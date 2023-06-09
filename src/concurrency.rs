pub trait Execute {
    fn execute<F>(&self, f: F) where F: FnOnce() + Send + 'static;
}

pub mod thread_pool;
