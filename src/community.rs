pub mod genome;
pub mod species;
pub mod recombination;
mod innovation_tracker;
mod mutation;

use crate::community::genome::Genome;
use crate::community::species::Species;
use crate::community::recombination::{Alignment, AlignmentParams};
use crate::neural_network::NeuralNetwork;

use innovation_tracker::InnovationTracker;
use serde::{Deserialize, Serialize};
use rstest::{rstest, fixture};

use self::genome::GenomeParams;


struct CommunityReproduction {
    genome_pool: Vec<Genome>,
    innovations: InnovationTracker,
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CommunityParams {
    pub species_thresh: f64,
    pub disjoint_importance: f64,
    pub excess_importance: f64,
    pub weight_importance: f64,
}

impl CommunityParams {
    pub fn new() -> CommunityParams {
        CommunityParams {
            species_thresh: 0.6,
            disjoint_importance: 1.,
            excess_importance: 1.,
            weight_importance: 0.2,
        }
    }
    
    pub fn get_test_params() -> CommunityParams {
        CommunityParams {
            species_thresh: 0.2,
            disjoint_importance: 1.,            
            excess_importance: 1.,
            weight_importance: 0.1,
        }
    }
}
const NUM_THREADS: usize = 2;

pub struct Community {
    pub genome_pool: Vec<Genome>,
    species: Vec<Species>,
    innovation_tracker: InnovationTracker,
    params: CommunityParams,
}

impl Community {
    // sorts and assigns genomes in the genome pool to species
    // kept seperate for testing
    // this means that community stores the unused genomes from construction
    pub fn identify_species(&self) -> Vec<Species> {

        // the returned vector
        let mut species_list = Vec::new();

        // each genome is used to make an organism of a particular genome
        for genome_i in 0..self.genome_pool.len() {

            // if we don't have any species so far we make one
            if species_list.len() == 0 {

                let mut new_species = Species::new(0);
                new_species.add_from_genome(self.genome_pool[genome_i].clone());
                species_list.push(new_species);
            
            // if we do have species we need to check them for compatibility
            } else {

                // check each species
                // this flag keeps track of if we need to make a new species or not
                let mut genome_assigned = false;
                for sp_i in 0..species_list.len() {

                    let incompatability = self.species_incompatibility(&self.genome_pool[genome_i], &species_list[sp_i]);

                    // if sufficiently compatible add to species, move on
                    if incompatability < self.params.species_thresh {
                        species_list[sp_i].add_from_genome(self.genome_pool[genome_i].clone());
                        genome_assigned = true;
                        break
                    }
                }

                // if the genome is left unassigned we 
                if !genome_assigned {
                    let mut new_species = Species::new(species_list.len());
                    new_species.add_from_genome(self.genome_pool[genome_i].clone());
                    species_list.push(new_species);

                }
            }
        }
        
        species_list
    }


    // used inside of previous to separate genomes into species
    fn species_incompatibility(&self, genome: &Genome, species: &Species) -> f64 {

        // pull out an example from the population
        let paragon = species.get_random_specimen();

        let a_params = AlignmentParams { crossover_mode: recombination::CrossoverMode::Alternating };
        let unweighted_incompatibility = Alignment::raw_incompatibility(genome, &paragon, &a_params);

        // correct the incompatibility values
        let disjoint = self.params.disjoint_importance * unweighted_incompatibility.base_disjoint;
        let excess = self.params.excess_importance * unweighted_incompatibility.base_excess;
        let weight_diff = self.params.weight_importance * unweighted_incompatibility.base_weight_diff;

        disjoint + excess + weight_diff
    }

    // calculates the species shared fitness to determine the sizes of the
    // populations of the species for the next generation
    fn next_species_sizes(&self, ffunc: fn(&NeuralNetwork) -> f64) -> Vec<usize> {
        
        // numerator and needed for denominator 
        let mut sums: Vec<f64> = Vec::new();
        let mut means: Vec<f64> = Vec::new();

        // for each species we need sum and mean fitness
        for species in &self.species {
            let fitness_vector = species.calculate_fitness(ffunc);
            let species_sum: f64 = fitness_vector.iter().sum();
            sums.push(species_sum);
            means.push(species_sum / (species.population.len() as f64));
        }

        // denominator is grand mean fitness
        let mean_fitness: f64 = means.iter().sum::<f64>() / (self.species.len() as f64);
        
        // get the actual next sizes
        let mut next_sizes = Vec::with_capacity(self.species.len());
        for fit_sum in sums {
            next_sizes.push((fit_sum / mean_fitness).round() as usize);
        }

        next_sizes
    }


    // for each species produce a new genome pool for the next generation of the community
    fn reproduce_all(&self, ffunc: fn(&NeuralNetwork) -> f64) -> CommunityReproduction {

        let mut new_innovation_tracker = self.innovation_tracker.clone();
        let target_sizes = self.next_species_sizes(ffunc);

        let mut total_new_genomes: usize = 0;

        for ts in &target_sizes {
            total_new_genomes += ts;
        }

        let mut new_genome_pool = Vec::with_capacity(total_new_genomes);

        for sp_i in 0..self.species.len() {
            let updates = self.species[sp_i].reproduce(target_sizes[sp_i], new_innovation_tracker);

            for genome in updates.population {
                new_genome_pool.push(genome);
            }

            new_innovation_tracker = updates.innovation_tracker;
        }

        CommunityReproduction {
            genome_pool: new_genome_pool,
            innovations: new_innovation_tracker,   
        }

    }

    pub fn generation(&mut self, fitness_function: fn(&NeuralNetwork) -> f64) -> Community {
        self.species = self.identify_species();
        let community_updates = self.reproduce_all(fitness_function);

        Community::from_reproduction(community_updates, self.params.clone())
    }

    pub fn get_all_fitness_values(&self, fitness_function: fn(&NeuralNetwork) -> f64) -> Vec<f64> {
        
        let mut raw_fitness = Vec::with_capacity(self.genome_pool.len());
        for genome in &self.genome_pool {
            raw_fitness.push(fitness_function(&genome.to_neural_network()));
        }

        raw_fitness
    }

    pub fn get_best_genome(&self, fitness_function: fn(&NeuralNetwork) -> f64) -> Genome {

        let fitness_values = self.get_all_fitness_values(fitness_function);

        let mut max_fit = 0.0;
        let mut max_index = 0;
        for (index, value) in fitness_values.iter().enumerate() {
            if value > &max_fit {
                max_fit = *value;
                max_index = index;
            }
        }

        self.genome_pool[max_index].clone()

    }

    pub fn new(n_genomes: usize, inputs: usize, outputs: usize) -> Community {
        
        // build genomes
        let mut genome_pool = Vec::with_capacity(n_genomes);
        for _ in 0..n_genomes {
            genome_pool.push(Genome::new_dense(inputs, outputs));
        }
        
        // find the highest innovation number
        let mut max_innov = 0;
        for genome in &genome_pool {
            for gene in &genome.edge_genes {
                if gene.innov > max_innov {
                    max_innov = gene.innov;
                }
            }
        }

        Community {
            genome_pool,
            species: Vec::new(),
            params: CommunityParams::new(),
            innovation_tracker: InnovationTracker::new(),
        }
    }

    fn from_reproduction(reproduction_state: CommunityReproduction, community_params: CommunityParams) -> Community {
        Community {
            genome_pool: reproduction_state.genome_pool,
            species: Vec::new(),
            params: community_params,
            innovation_tracker: reproduction_state.innovations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genome::GenomeParams;

    #[fixture]
    fn test_community() -> Community {
        // make a couple of sparse genomes
        let mut gen_0 = Genome::new_dense(3, 4);
        gen_0._remove_by_innovation(0);
        gen_0._remove_by_innovation(1);
        gen_0._remove_by_innovation(2);
        gen_0.params = GenomeParams::get_test_params();

        let mut gen_1 = Genome::new_dense(3, 4);
        gen_1._remove_by_innovation(0);
        gen_1._remove_by_innovation(1);
        gen_1._remove_by_innovation(2);
        gen_1.params = GenomeParams::get_test_params();

        // make a couple of dense genomes
        let mut gen_2 = Genome::new_dense(3, 4);
        gen_2.params = GenomeParams::get_test_params();
        let mut gen_3 = Genome::new_dense(3, 4);
        gen_3.params = GenomeParams::get_test_params();

        // make community and assign species
        let mut comm = Community::new(4, 3, 4);
        
        // insert our knowns
        comm.params = CommunityParams::get_test_params();
        comm.genome_pool[0] = gen_0.clone();
        comm.genome_pool[1] = gen_1.clone();
        comm.genome_pool[2] = gen_2.clone();
        comm.genome_pool[3] = gen_3.clone();
        
        comm

    }

    #[rstest]
    fn test_identify_species(test_community: Community) {

        let mut comm = test_community;

        comm.species = comm.identify_species();

        println!("Number of species: {}", comm.species.len());

        for i in 0..comm.species.len() {
            println!("Population size of species {}: {}", i, comm.species[i].size)
        }

        assert!(comm.species[0].population[0] == comm.genome_pool[0]);
        assert!(comm.species[0].population[1] == comm.genome_pool[1]);
        assert!(comm.species[1].population[0] == comm.genome_pool[2]);
        assert!(comm.species[1].population[1] == comm.genome_pool[3]);
        assert!(comm.species.len() == 2) 
    }

    #[rstest]
    fn test_next_species_sizes() {
        // make a sparse genome
        let mut gen_1 = Genome::new_dense(2, 3);
        gen_1._remove_by_innovation(0);
        gen_1._remove_by_innovation(1);
        gen_1._remove_by_innovation(2);
        gen_1._remove_by_innovation(4);
        
        let mut gen_4 = Genome::new_dense(2, 3);
        gen_4._remove_by_innovation(0);
        gen_4._remove_by_innovation(1);
        gen_4._remove_by_innovation(2);

        gen_1.params = GenomeParams::get_test_params();

        // make a couple of dense genomes
        let mut gen_2 = Genome::new_dense(2, 3);
        gen_2.params = GenomeParams::get_test_params();
        let mut gen_3 = Genome::new_dense(2, 3);
        gen_3.params = GenomeParams::get_test_params();

        // make community and assign species
        let mut comm = Community::new(4, 2, 3);
        
        // insert our knowns
        comm.params = CommunityParams::get_test_params();
        comm.genome_pool[0] = gen_1.clone();
        comm.genome_pool[1] = gen_2.clone();
        comm.genome_pool[2] = gen_3.clone();
        comm.genome_pool[3] = gen_4.clone();
        
        comm.species = comm.identify_species();
        println!{"{:?}", comm.species.len()};

        // define a simple fitness function
        fn genome_len(nn: &NeuralNetwork) -> f64 {
            nn.edges.len() as f64
        }

        // get the sizes
        let sizes = comm.next_species_sizes(genome_len);

        println!("{:?}", sizes);
        assert_eq!(sizes[0], 1);
        assert_eq!(sizes[1], 3)
    }

    #[rstest]
    fn test_reproduce_all_count() {

    }
}