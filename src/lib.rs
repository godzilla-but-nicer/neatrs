//#![warn(missing_docs)]
#![allow(unused_imports, dead_code, unused_variables)]

//! its NEAT, baby

/// This module contains all of the neural network stuff
pub mod neural_network;

// This module contains all of the evolutionary stuff
pub mod community;

use community::genome::{Genome, GenomeParams};
use community::{Community, CommunityParams};
use community::species::SpeciesParams;  // param structs should be moved up to community for public
use community::recombination::CrossoverMode;

use neural_network::NeuralNetwork;
use serde::{Serialize, Deserialize};
use rstest::{rstest, fixture};

use std::fs::{self, File};
use std::io::{prelude::*, BufWriter};

#[derive(Serialize, Deserialize)]
pub struct NEATParams {
    num_individuals: usize,
    num_inputs: usize,
    num_outputs: usize,
    community: CommunityParams,
    species: SpeciesParams,
    genome: GenomeParams,
}

pub struct NEAT {
    community: Community,
    fitness_function: fn(&NeuralNetwork) -> f64,
    params: NEATParams,
}

impl NEAT {
    
    pub fn evolve(&mut self, num_generations: usize) {

        for generation in 0..num_generations {
            let next_community;
            next_community = self.community.generation(self.fitness_function);
            self.community = next_community;
        }

    }


    pub fn from_parameters(parameter_path: &str, fitness: fn(&NeuralNetwork) -> f64) -> NEAT {

        let neat_parameters = Self::read_parameter_file(parameter_path);
        
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

    pub fn write_fitness_values(&self, path: &str) {

        let fitness_values = self.community.get_all_fitness_values(self.fitness_function);

        let mut buffer = File::create(path).unwrap();
        let mut writer = BufWriter::new(buffer);
        for value  in fitness_values {
            let line = format!("{value}\n");
            writeln!(&mut writer, "{}", value).unwrap();
        }

    }

    pub fn get_champion(&self) -> Genome {
        self.community.get_best_genome(self.fitness_function)
    }


    fn read_parameter_file(path_string: &str) -> NEATParams {

        // extract a string from the file
        let yaml_string = match fs::read_to_string(&path_string) {
            Err(e) => panic!("Unable to open {path_string}: {e}"),
            Ok(s) => s,
        };

        Self::parse_parameter_yaml(&yaml_string)

    }

    fn parse_parameter_yaml(fstring: &str) -> NEATParams {

        let params: NEATParams = serde_yml::from_str(fstring).unwrap();

        params
    }

    pub fn number_of_genomes(&self) -> usize {
        self.community.genome_pool.len()
    } 

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NEATParams;


    #[rstest]
    fn test_parse_parameter_string() {
        let pstring = r#"
        num_individuals: 3
        num_inputs: 2
        num_outputs: 1

        community: 
          species_thresh: 0.6
          disjoint_importance: 1.0
          excess_importance: 1.0
          weight_importance: 1.0

        species:
          mate_fraction: 0.7
          num_elites: 1
          prob_structural: 0.3
          crossover_mode: SimpleRandom

        genome:
          insert_prob: 0.5
          connect_prob: 0.5
          weight_mut_sd: 1.0
          bias_mut_sd: 1.0
        "#;


        let t: NEATParams = NEAT::parse_parameter_yaml(pstring);

        assert_eq!(t.community.excess_importance, 1.0)
    }

}
