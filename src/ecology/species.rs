use crate::ecology::organism::Organism;
use crate::ecology::genome::Genome;
use rand::prelude::*;

pub struct Species {
    id: usize,
    population: Vec<Organism>,
}

impl Species {
    // adds an organism to the population
    pub fn add_from_genome(&mut self, gen: Genome) {
        let new_org = Organism::new(gen);
        self.population.push(new_org);
    }

    // produce a random member to check compatibility
    pub fn get_random_specimen(&self) -> &Organism {
        let mut rng = rand::thread_rng();
        let specimen_i = rng.gen_range(0..self.population.len());
        &self.population[specimen_i]
    }


    pub fn new(id: usize) -> Species {
        Species {
            id,
            population: Vec::new(),
        }
    }
}