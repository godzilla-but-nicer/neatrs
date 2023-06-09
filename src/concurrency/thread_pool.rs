use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use crate::concurrency::Execute;

type Task = Box<dyn FnOnce() + Send + 'static>;

struct Worker {
    _id: usize,
    _thread: thread::JoinHandle<()>,
}

impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<Receiver<Task>>>) -> Self {
        let thread_name = format!("worker-{}", id);
        let _thread = thread::Builder::new()
            .name(thread_name)
            .spawn(move || loop {

                let task = receiver.lock().unwrap().recv().unwrap();
                task();
            })
            .unwrap();
        Self { _id: id, _thread }
    }
}

pub struct ThreadPool {
    _workers: Vec<Worker>,
    task_sender: Sender<Task>,
}

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        assert!(size > 0);
        let (task_sender, task_receiver) = channel();
        let task_receiver = Arc::new(Mutex::new(task_receiver));
        let mut _workers = Vec::with_capacity(size);
        for id in 0..size {
            _workers.push(Worker::new(id, Arc::clone(&task_receiver)));
        }
        Self {
            _workers,
            task_sender,
        }
    }
}

impl Execute for ThreadPool {
    fn execute<F>(&self, f: F) 
    where
        F: FnOnce() + Send + 'static,
    {
        self.task_sender.send(Box::new(f)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::super::Execute;
    use super::ThreadPool;
    use std::sync::mpsc::channel;
    use std::thread::sleep;
    use std::time;

    #[test]
    fn test_simple() {
        let tp = ThreadPool::new(4);
        let (tx, rx) = channel();

        for i in 0..10 {
            let tx = tx.clone();
            tp.execute(move || {
                tx.send(i).expect("send failed");
            });
        }

        // sum([0, 1, 2, ..., 9] == 45)
        assert_eq!(rx.iter().take(10).sum::<usize>(), 45);

    }

    #[test]
    fn test_concurrency() {
        let task_duration = time::Duration::from_millis(500);
        let (tx, rx) = channel();

        // 2 threads for 2 tasks
        let tp = ThreadPool::new(2);

        let time_start = time::Instant::now();

        for i in 0..2 {
            let tx = tx.clone();
            tp.execute(move || {
                sleep(task_duration);
                tx.send(i).expect("send failed");
            });
        }

        // consume both results
        rx.iter().take(2).for_each(|_| {});

        // The only way for this to be true is if the tasks were executed concurrently
        assert!(time_start.elapsed() < 2 * task_duration);
    }

    #[test]
    fn test_limited_concurrency() {
        let task_duration = time::Duration::from_millis(500);
        let (tx, rx) = channel();

        // 1 thread for 2 tasks
        let tp = ThreadPool::new(1);

        let time_start = time::Instant::now();

        for i in 0..2 {
            let tx = tx.clone();
            tp.execute(move || {
                sleep(task_duration);
                tx.send(i).expect("send failed");
            });
        }

        // consume both results
        rx.iter().take(2).for_each(|_| {});

        // This should only be true if the tasks were *not* executed concurrently
        assert!(time_start.elapsed() >= 2 * task_duration);
    }
}
