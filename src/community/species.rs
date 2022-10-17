use crate::community::organism::Organism;
use crate::community::genome::Genome;
use rand::prelude::*;

use crossbeam::thread;

const NUM_THREADS: usize = 6;

#[derive(Debug)]
pub struct Species {
    id: usize,
    pub population: Vec<Organism>,
    next_size: usize,
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

    // calculates the raw fitness value that must be adjusted before doing any
    // evolution with it
    // should probably be broken up. e.g. spawn_fitness_thread() or spawn chunk_threads()
    fn calculate_raw_fitness(&self, ffunc: fn(&Organism) -> f64) -> Vec<f64> {

        // we're going to recieve indices and values from the child threads
        let mut i_fit_pairs = Vec::new();

        // for multithreading we need to calculate fitness in chunks
        let chunks = self.population.len() / NUM_THREADS;
        let remainder = self.population.len() % NUM_THREADS;
        
        for chunk_i in 0..chunks {
            
            let chunk_start = chunk_i * NUM_THREADS;
            
            // scope per chunk in which all multithreading will occur
            let s = thread::scope(|s| {

                // this vector contains the data passed from child threads
                let mut handles: Vec<thread::ScopedJoinHandle<(usize, f64)>> = Vec::new();
                for org_j in 0..NUM_THREADS {

                    // add handles for organism index and fitness from thread
                    handles.push(s.spawn(move |_| {
                        let raw_fitness = ffunc(&self.population[org_j]);
                        (chunk_start + org_j, raw_fitness)
                    }));
                }

                // move the handle data into a return vector for the scope
                let mut chunk_data = Vec::new();
                for handle in handles {
                    chunk_data.push(handle.join().unwrap());
                }
                chunk_data
            });
            
            // add data from scoped threads to main thread vector
            i_fit_pairs.append(&mut s.unwrap());
        }

        
        // now we have to calculate the ending partial chunk
        let remain_start = i_fit_pairs.len();

        // this is the same as the inner loop above
        let s = thread::scope(|s| {
            let mut handles: Vec<thread::ScopedJoinHandle<(usize, f64)>> = Vec::new();
            for org_j in 0..remainder {
                handles.push(s.spawn(move |_| {
                    let raw_fitness = ffunc(&self.population[org_j]);
                    (remain_start + org_j, raw_fitness)
                }));
            }

            // unpack the handles
            let mut remain_data = Vec::new();
            for handle in handles {
                remain_data.push(handle.join().unwrap());
            }
            remain_data
        });

        // add remaining values
        i_fit_pairs.append(&mut s.unwrap());

        // we will use the data from the scope to fill the ordered vector
        let mut raw_fitness = vec![0.0; self.population.len()];
        for (org_j, fitness) in &i_fit_pairs {
            raw_fitness[*org_j] = *fitness;
        }

        raw_fitness
    }

    // returns the adjusted fitness values for each individual
    pub fn calculate_fitness(&self, ffunc: fn(&Organism) -> f64) -> Vec<f64> {

        let raw_fitness = self.calculate_raw_fitness(ffunc);
        let mut adj_fitness = Vec::with_capacity(raw_fitness.len());

        // normalize by population size
        for raw_fit in raw_fitness {
            adj_fitness.push(raw_fit / (self.population.len() as f64));
        }

        adj_fitness
    }

    pub fn new(id: usize) -> Species {
        Species {
            id,
            population: Vec::new(),
            next_size: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_calculate_raw_fitness() {

        fn n_genes(org: &Organism) -> f64 {
            org.genome.edge_genes.len() as f64
        }
        
        // genome size is easy test fitness func
        let gen_1 = Genome::new_dense(2, 1);
        let gen_2 = Genome::new_dense(3, 1);
        let gen_3 = Genome::new_dense(3, 2);
        
        let mut test_species = Species::new(0);
        test_species.add_from_genome(gen_1);
        test_species.add_from_genome(gen_2);
        test_species.add_from_genome(gen_3);
        
        let fitness_vals = test_species.calculate_raw_fitness(n_genes);

        assert!((fitness_vals[0] - 2.0).abs() < 1e5);
        assert!((fitness_vals[1] - 3.0).abs() < 1e5);
        assert!((fitness_vals[2] - 6.0).abs() < 1e5)
    }

}