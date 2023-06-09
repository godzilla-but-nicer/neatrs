use crate::community::genome::Genome;

#[derive(Debug)]
pub struct Organism {
    pub genome: Genome,
    pub raw_fitness: Option<f64>,
}

impl Organism {
    pub fn new(genome: Genome) -> Organism {
        Organism {
            genome,
            raw_fitness: None,
        }
    }

    pub fn get_fitness(&self) -> Option<f64> {
        self.raw_fitness
    }

    pub fn set_fitness(&mut self, fitness: f64) {
        self.raw_fitness = Some(fitness);
    }
}
