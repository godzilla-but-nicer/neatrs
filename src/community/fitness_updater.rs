use super::Organism;
use crate::concurrency::Execute;
use std::sync::mpsc::{channel, Receiver, Sender};

// Utility struct for updating a vector of organisms' fitnesses according
// to a fitness function. Concurrency and parallelism can be achieved by using a
// specified executor `T` to run the fitness function on each organism.
pub struct FitnessUpdater<T>
where
    T: Execute,
{
    executor: T,
    sender: Sender<(usize, f64)>,
    receiver: Receiver<(usize, f64)>,
}

impl<T> FitnessUpdater<T>
where
    T: Execute,
{
    // Creates a new FitnessUpdater with the specified executor.
    pub fn new(executor: T) -> Self {
        let (sender, receiver) = channel();
        Self {
            executor,
            sender,
            receiver,
        }
    }

    // Updates the fitness of each organism in the vector according to the
    // specified fitness function. This may run concurrently depending on the
    // executor used, but calling this function will block until all organisms
    // have been updated.
    pub fn update_fitness(&self, mut orgs: &Vec<Organism>, fitness_func: fn(&Organism) -> f64) {
        orgs.iter().enumerate().for_each(|(i, org)| {
            let sender = self.sender.clone();
            self.executor.execute(move || {
                let fitness = fitness_func(&org);
                sender.send((i, fitness)).unwrap();
            });
        });
        self.receiver
            .iter()
            .take(orgs.len())
            .for_each(|(i, fitness)| {
                orgs[i].raw_fitness = Some(fitness);
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};

    // Need this to construct organisms
    use crate::community::genome;

    #[fixture]
    fn orgs_ascending_fitness(#[default(0)] num: usize) -> Vec<Organism> {
        (0..num)
            .into_iter()
            .map(|i| {
                let genome = genome::Genome::new_dense(1, 1);
                let mut org = Organism::new(genome);
                org.raw_fitness = Some(i as f64);
                org
            })
            .collect()
    }

    #[fixture]
    fn ffunc_negate() -> fn(&Organism) -> f64 {
        |org: &Organism| -org.get_fitness().unwrap()
    }

    #[fixture]
    fn ffunc_ones() -> fn(&Organism) -> f64 {
        |_: &Organism| 1.0
    }

    #[fixture]
    fn mock_executor() -> impl Execute {
        struct MockExecutor {
            sender: Sender<()>,
        }
        impl Execute for MockExecutor {
            fn execute<F>(&self, f: F)
            where
                F: FnOnce() + Send + 'static,
            {
                f();
                self.sender.send(()).unwrap();
            }
        }
        let (sender, _) = channel();
        MockExecutor { sender }
    }

    #[rstest]
    fn test_empty(
        mock_executor: impl Execute,
        #[with(0)] #[from(orgs_ascending_fitness)] orgs: Vec<Organism>,
        ffunc_ones: fn(&Organism) -> f64,
    ) {
        let updater = FitnessUpdater::new(crate::concurrency::thread_pool::ThreadPool::new(4));
        let orgs = vec![];
        updater.update_fitness(&orgs, ffunc_ones);
        assert_eq!(orgs.len(), 0);
    }

    #[rstest]
    fn test_single(
        mock_executor: impl Execute,
        #[with(1)] #[from(orgs_ascending_fitness)] orgs: Vec<Organism>,
        ffunc_ones: fn(&Organism) -> f64,
    ) {
        let updater = FitnessUpdater::new(mock_executor);
        updater.update_fitness(&orgs, ffunc_ones);
        assert_eq!(orgs.len(), 1);
        assert_eq!(orgs[0].get_fitness().unwrap(), 1.0);
    }

    #[rstest]
    fn test_multiple_ordering_matters(
        mock_executor: impl Execute,
        #[with(3)] #[from(orgs_ascending_fitness)] orgs: Vec<Organism>,
        ffunc_negate: fn(&Organism) -> f64,
    ) {
        let updater = FitnessUpdater::new(mock_executor);
        updater.update_fitness(&orgs, ffunc_negate);

        assert_eq!(orgs.len(), 3);
        assert_eq!(orgs[0].get_fitness().unwrap(), -0.0);
        assert_eq!(orgs[1].get_fitness().unwrap(), -1.0);
        assert_eq!(orgs[2].get_fitness().unwrap(), -2.0);
    }
}
