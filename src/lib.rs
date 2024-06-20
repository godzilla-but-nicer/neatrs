#![warn(missing_docs)]

//! its NEAT, baby

/// This module contains all of the neural network stuff
pub mod neural_network;

// This modual contains all of the evolutionary stuff
mod community;

use community::genome::Genome;
use community::{Community, CommunityParams};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct NEATParams {
    num_individuals: usize,
    num_inputs: usize,
    num_outputs: usize,
    community: CommunityParams,
    species: SpeciesParams,
    genome: GenomeParams,
}

pub struct NEAT {
    community: Community,
    fitness_function: fn(Genome) -> f64,
    params: NEATParams,
}

impl NEAT {
    pub fn from_parameters(neat_parameters: NEATParams, fitness: fn(Genome) -> f64) -> NEAT {
        
        let new_community = Community::new(
            neat_parameters.num_individuals,
            neat_parameters.num_inputs,
            neat_parameters.num_outputs,
        );

        NEAT {
            community: new_community,
            fitness_function: fitness,
            params: neat_parameters,
        }
    }

    pub fn read_parameter_file(path: &str) -> NEATParams {

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
