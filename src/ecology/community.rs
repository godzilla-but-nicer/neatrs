use crate::ecology::genome::Genome;
use crate::ecology::species::Species;
use crate::ecology::community_params::CommunityParams;

pub struct Community {
    genome_pool: Vec<Genome>,
    species: Vec<Species>,
    next_innovation: usize,
    params: CommunityParams,
}

impl Community {
    fn identify_species(&mut self) {
        let mut species_id = self.species.len();

        // each genome is used to make an organism of a particular genome
        for genome in &self.genome_pool {

            // if we don't have any species so far we make one
            if self.species.len() == 0 {
                let mut new_species = Species::new(species_id);
                new_species.add_from_genome(genome.clone());
                self.species.push(new_species);
                species_id += 1;
            
            // if we do have species we need to check them for compatibility
            } else {

                // check each species
                for sp_i in 0..self. species.len() {

                    let paragon = self.species[sp_i].get_random_specimen();
                    let incomp = genome.incompatibility(&paragon.genome, 
                                                        self.params.disjoint_imp, 
                                                        self.params.excess_imp, 
                                                        self.params.weight_imp, 
                                                        self.next_innovation);
                    // if sufficiently compatible add to species
                    if incomp < self.params.species_thresh {
                        self.species[sp_i].add_from_genome(genome.clone());
                        break
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