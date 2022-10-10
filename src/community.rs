mod community_params;
pub mod genome;
mod organism;
mod species;

use crate::community::genome::Genome;
use crate::community::species::Species;
use crate::community::community_params::CommunityParams;

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
    fn identify_species(&mut self) {

        // each genome is used to make an organism of a particular genome
        for genome in &self.genome_pool {

            // if we don't have any species so far we make one
            if self.species.len() == 0 {
                let mut new_species = Species::new(0);
                new_species.add_from_genome(genome.clone());
                self.species.push(new_species);
            
            // if we do have species we need to check them for compatibility
            } else {

                // check each species
                for sp_i in 0..self. species.len() {

                    let paragon = self.species[sp_i].get_random_specimen();
                    let incomp = genome.incompatibility(&paragon.genome, 
                                                        self.next_innovation);
                    // if sufficiently compatible add to species, move on
                    if incomp < self.params.species_thresh {
                        self.species[sp_i].add_from_genome(genome.clone());
                        break

                    // make new species if needed
                    } else if sp_i == (self.species.len() - 1) {
                        let mut new_species = Species::new(sp_i + 1);
                        new_species.add_from_genome(genome.clone());
                        self.species.push(new_species);
                    }
                }
            }
        }
    }

    fn new(n_genomes: usize, inputs: usize, outputs: usize) -> Community {
        
        // build genomes
        let mut genome_pool = Vec::with_capacity(n_genomes);
        for _ in 0..n_genomes {
            genome_pool.push(Genome::new_minimal_dense(inputs, outputs));
        }
        
        // find the highest innovation number
        let mut max_innov = 0;
        for genome in &genome_pool {
            for gene in &genome.edge_genes {
                if gene.innovation > max_innov {
                    max_innov = gene.innovation;
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
    use crate::community::community_params::*;

    #[test]
    fn test_identify_species() {
        // make a sparse genome
        let mut gen_1 = Genome::new_minimal_dense(2, 3);
        gen_1._remove_by_innovation(0);
        gen_1._remove_by_innovation(1);
        gen_1._remove_by_innovation(2);
        gen_1._remove_by_innovation(4);

        gen_1.params = GenomeParams::get_test_params();

        // make a couple of dense genomes
        let mut gen_2 = Genome::new_minimal_dense(2, 3);
        gen_2.params = GenomeParams::get_test_params();
        let mut gen_3 = Genome::new_minimal_dense(2, 3);
        gen_3.params = GenomeParams::get_test_params();

        // make community and assign species
        let mut comm = Community::new(3, 2, 3);
        
        // insert our knowns
        comm.params = CommunityParams::get_test_params();
        comm.genome_pool[0] = gen_1.clone();
        comm.genome_pool[1] = gen_2.clone();
        comm.genome_pool[2] = gen_3.clone();
        
        comm.identify_species();

        println!("{}", comm.species.len());
        assert!(comm.species[0].population[0].genome == gen_1);
        assert!(comm.species[1].population[0].genome == gen_2);
        assert!(comm.species[1].population[1].genome == gen_3);
        assert!(comm.species.len() == 2) 
    }
}