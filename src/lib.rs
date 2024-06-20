#![warn(missing_docs)]

//! its NEAT, baby

/// This module contains all of the neural network stuff
pub mod neural_network;

// This modual contains all of the evolutionary stuff
mod community;

use community::Community;
use community::genome::Genome;

struct NEATParams {
    community_size: usize,
}

pub struct NEAT {
    community: Community,
    fitness_function: fn(Genome) -> f64,
    params: NEATParams,
}

impl NEAT {
    pub fn new() -> NEAT {
        
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
