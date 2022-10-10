use crate::community::genome::Genome;

pub struct Organism {
    pub genome: Genome,
    pub raw_fitness: f64,
}

impl Organism {
    pub fn new(genome: Genome) -> Organism {
        Organism {
            genome,
            raw_fitness: -1.0,
        }
    }
}