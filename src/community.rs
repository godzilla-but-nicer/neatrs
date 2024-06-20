pub mod genome;
mod organism;
mod species;
mod recombination;
mod innovation_tracker;
mod mutation;

use crate::community::genome::Genome;
use crate::community::species::Species;
use crate::community::organism::Organism;
use crate::community::recombination::{Alignment, AlignmentParams};

use serde::{Deserialize, Serialize};

use self::genome::GenomeParams;


#[derive(Serialize, Deserialize, Debug)]
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
    genome_pool: Vec<Genome>,
    species: Vec<Species>,
    next_innovation: usize,
    params: CommunityParams,
}

impl Community {
    // sorts and assigns genomes in the genome pool to species
    // kept seperate for testing
    // this means that community stores the unused genomes from construction
    fn identify_species(&self) -> Vec<Species> {

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

    fn species_incompatibility(&self, genome: &Genome, species: &Species) -> f64 {

        // pull out an example from the population
        let paragon = species.get_random_specimen();

        let a_params = AlignmentParams { crossover_mode: recombination::CrossoverMode::Alternating };
        let unweighted_incompatibility = Alignment::raw_incompatibility(genome, &paragon.genome, &a_params);

        // correct the incompatibility values
        let disjoint = self.params.disjoint_importance * unweighted_incompatibility.base_disjoint;
        let excess = self.params.excess_importance * unweighted_incompatibility.base_excess;
        let weight_diff = self.params.weight_importance * unweighted_incompatibility.base_weight_diff;

        disjoint + excess + weight_diff
    
    }

    // calculates the species shared fitness to determine the sizes of the
    // populations of the species for the next generation
    fn next_species_sizes(&self, ffunc: fn(&Organism) -> f64) -> Vec<usize> {
        
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
            next_innovation: max_innov + 1,
            params: CommunityParams::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genome::GenomeParams;

    #[test]
    fn test_identify_species() {
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
        
        comm.species = comm.identify_species();

        println!("Number of species: {}", comm.species.len());

        for i in 0..comm.species.len() {
            println!("Population size of species {}: {}", i, comm.species[i].size)
        }

        assert!(comm.species[0].population[0].genome == gen_0);
        assert!(comm.species[0].population[1].genome == gen_1);
        assert!(comm.species[1].population[0].genome == gen_2);
        assert!(comm.species[1].population[1].genome == gen_3);
        assert!(comm.species.len() == 2) 
    }

    #[test]
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
        fn genome_len(org: &Organism) -> f64 {
            org.genome.edge_genes.len() as f64
        }

        // get the sizes
        let sizes = comm.next_species_sizes(genome_len);

        println!("{:?}", sizes);
        assert_eq!(sizes[0], 1);
        assert_eq!(sizes[1], 3)
    }
}