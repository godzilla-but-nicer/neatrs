use crate::ecology::genome::Genome;
use crate::ecology::species::Species;

pub struct Community {
    genome_pool: Vec<Genome>,
    species: Vec<Species>,
}

impl Community {
    fn identify_species(&self) {
        let mut species_id = 0;

        // each genome is used to make an organism of a particular genome
        for genome in &self.genome_pool {

            // if we don't have any species so far we make one
            if self.species.len() == 0 {
                let mut new_species = Species::new(species_id);
                new_species.add_from_genome(genome.clone());
                species_id += 1;
            
            // if we do have species we need to check them for compatibility
            } else {

            }
        }
    }
}