use crate::community::genome::Genome;
use crate::community::recombination::{Alignment, AlignmentParams, CrossoverMode};
use crate::neural_network::edge::Edge;
use crate::community::mutation::{mutate};

use rand::prelude::*;
use crossbeam::thread;

use serde::{Serialize, Deserialize};

use super::innovation_tracker::{self, InnovationTracker};
use super::mutation::MutationInfo;

const NUM_THREADS: usize = 2;

/// Provides named fields for parameters that specify how species and reproduction function.
/// 
/// # Attributes
/// 
/// * `mate_fraction` - proportion of the highest fitness individuals who get to reproduce in the
///   current round
/// * `num_elites` - Number of individuals who get to pass along their genomes without the process
///   of sexual reproduction
/// * `prob_structural` - The probability of a structural mutation occurring during reproduction
/// * `crossover_mode` - The type of crossing other that occurs when recombining a a child genome
#[derive(Serialize, Deserialize, Debug)]
pub struct SpeciesParams {
    pub mate_fraction: f64,
    pub num_elites: usize,
    pub prob_structural: f64,
    pub crossover_mode: CrossoverMode,
}

impl SpeciesParams {
    pub fn new() -> SpeciesParams {
        SpeciesParams {
            mate_fraction: 0.70,
            num_elites: 3,
            prob_structural: 0.3,
            crossover_mode: CrossoverMode::SimpleRandom,
        }
    }

    pub fn get_test_params() -> SpeciesParams {
        SpeciesParams {
            mate_fraction: 0.70,
            num_elites: 1,
            prob_structural: 0.3,
            crossover_mode: CrossoverMode::Alternating,
        }
    }
}

pub struct ReproductionUpdates {
    pub population: Vec<Genome>,
    pub innovation_tracker: InnovationTracker,
}

/// The Species struct holds all of the data for sorting the population of genomes into subgroups
/// for reproduction along with methods for handling reproduction.
/// 
/// # Attributes
/// 
/// * `id` - Number identifying the species
/// * `size` - Number of individuals in the species
/// * `population` - Vector of Organisms making up the species
/// * `params` - Struct holding the relevant parameters
#[derive(Debug)]
pub struct Species {
    id: usize,
    pub size: usize,
    pub population: Vec<Genome>,
    params: SpeciesParams, 
}

impl Species {
    /// Add an Organism to the population from its Genome
    pub fn add_from_genome(&mut self, gen: Genome) {
        self.population.push(gen);
        self.size += 1;
    }

    /// Produces a random member of the Species used to check compatability when populating species.
    pub fn get_random_specimen(&self) -> &Genome {
        let mut rng = rand::thread_rng();
        let specimen_i = rng.gen_range(0..self.size);
        &self.population[specimen_i]
    }

    /// Produces a number of new genomes by passing along elite genome and pulling random pairs of
    /// high fitness individuals for sexual reproduction.
    /// 
    /// # Arguments
    /// 
    /// * `target_size` - Size of the new population resulting from the reproduction within this
    ///   species
    /// * `innovation_tracker` - Holds information about the highest discovered innovation number
    pub fn reproduce(&self, target_size: usize, mut innovation_tracker: InnovationTracker) -> ReproductionUpdates {        

        // determine who reproduces and who passes unchanged
        let mut reproducers = self.get_reproducers();
        let elites = self.get_elites(self.params.num_elites);
        let mut new_population: Vec<Genome> = Vec::new();

        // elites pass down directly with no mutation
        for elite_i in elites {
            let new_elite = self.population[elite_i].clone();
            new_population.push(new_elite);
        }
        
        // initialize rng for random mating
        let mut rng = rand::thread_rng();
        
        // each non-elite organism is generated by mating of two other organisms
        for _ in self.params.num_elites..target_size {
            
            // we need a random pairing
            reproducers.shuffle(&mut rng);
            let mom = self.population[reproducers[0]].clone();
            let dad = self.population[reproducers[1]].clone();
            let parents = [&mom, &dad];
            
            // align genomes and crossover into new genome
            let params = AlignmentParams { crossover_mode: self.params.crossover_mode.clone() };
            let mut crossed_genome = Alignment::crossover(mom, dad, &params);
            
            // mutate the new genomes
            let mut_info;
            if thread_rng().gen::<f64>() < self.params.prob_structural {
                mut_info = mutate(crossed_genome, &innovation_tracker, true);
            } else {
                mut_info = mutate(crossed_genome, &innovation_tracker, false);
            }

            // unpack the mutation info and update the tracker where appropriate
            innovation_tracker = innovation_tracker.update(&mut_info);
            match mut_info {
                MutationInfo::Quantitative(gen) => new_population.push(gen),
                MutationInfo::Topological((gen, _)) => new_population.push(gen),
            };
        }
        
        ReproductionUpdates { 
            population: new_population, 
            innovation_tracker
        }

    }

    /// Elites are the individuals with fitness high enough that their genomes pass directly down
    /// to their descendants without recombination
    fn get_elites(&self, num_elites: usize) -> Vec<usize> {

        // get sorted indices
        let fit_idx = self.sorted_pop_indices();

        // slice out the elite indices
        fit_idx[0..self.params.num_elites].to_vec()
    }

    /// Determine indices of reproducing organisms by sortinf by fitness and taking the fraction
    /// described in the `mate_fraction` parameter.
    fn get_reproducers(&self) -> Vec<usize> {

        // sort population indices by fitness
        let fit_idx = self.sorted_pop_indices();

        // get the top fraction given in the config
        let mate_float = (self.size as f64) * self.params.mate_fraction;
        let mate_number = mate_float.round() as usize;
        
        // return the indices that get to reproduce
        fit_idx[0..mate_number].to_vec()
    }

    /// Sort the Organisms in the species by fitness in descending order. Needed for getting elites
    /// and reproducers
    fn sorted_pop_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.size).collect();

        // this long statement does the actual sorting
        indices.sort_unstable_by(|a, b| 
            self.population[*b].raw_fitness.partial_cmp(
            &self.population[*a].raw_fitness)
            .unwrap());

        indices
    }


    /// Calculates the raw fitness value for each Organism in the population. This is where the
    /// heaviest computation occurs in evaluating the fitness function. We will adjust these values
    /// before using them to make evolutionary comparisons. This function should probably be broken
    /// up. e.g. spawn_fitness_thread() or spawn chunk_threads()
    fn calculate_raw_fitness(&self, ffunc: fn(&Genome) -> f64) -> Vec<f64> {

        // we're going to recieve indices and values from the child threads
        let mut i_fit_pairs = Vec::new();

        // for multithreading we need to calculate fitness in chunks
        let chunks = self.size / NUM_THREADS;
        let remainder = self.size % NUM_THREADS;
        
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
        let mut raw_fitness = vec![0.0; self.size];
        for (org_j, fitness) in &i_fit_pairs {
            raw_fitness[*org_j] = *fitness;
        }

        raw_fitness
    }

    /// This function returns the adjusted fitness values for each genome. These are the values we
    /// use to compare species.
    pub fn calculate_fitness(&self, ffunc: fn(&Genome) -> f64) -> Vec<f64> {

        let raw_fitness = self.calculate_raw_fitness(ffunc);
        let mut adj_fitness = Vec::with_capacity(raw_fitness.len());

        // normalize by population size
        for raw_fit in raw_fitness {
            adj_fitness.push(raw_fit / (self.size as f64));
        }

        adj_fitness
    }

    /// Default constructor but shouldn't be. We want our defualt constructor function signatures to
    /// look right with no arguments. For now this takes an `id` value to label the species and
    /// makes an empty Species 
    pub fn new(id: usize) -> Species {
        Species {
            id,
            size: 0,
            population: Vec::new(),
            params: SpeciesParams::new(),
        }
    }
    
    /// This function is used to make species for tests.
    pub fn _add_genome(&mut self, org: Genome) {
        self.population.push(org);
        self.size += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::community::recombination::Alignment;

    use super::*;

    #[test]
    fn test_sorted_pop_fitness() {
        // built a species with some organisms
        let mut spe = Species::new(0);
        spe._add_genome(Genome::new_dense(2, 2));
        spe._add_genome(Genome::new_dense(2, 2));
        spe._add_genome(Genome::new_dense(2, 2));

        // assign some arbitrary fitness values
        spe.population[1].raw_fitness = 100.0;
        spe.population[2].raw_fitness = 10.0;
        spe.population[0].raw_fitness = 1.0;

        let sorted_idx = spe.sorted_pop_indices();

        assert_eq!(vec![1, 2, 0], sorted_idx);
    }

    #[test]
    fn test_get_reproducers() {
        // built a species with some organisms
        let mut spe = Species::new(0);
        spe.params = SpeciesParams::get_test_params();
        spe._add_genome(Genome::new_dense(2, 2));
        spe._add_genome(Genome::new_dense(2, 2));
        spe._add_genome(Genome::new_dense(2, 2));

        // assign some arbitrary fitness values
        spe.population[1].raw_fitness = 100.0;
        spe.population[2].raw_fitness = 10.0;
        spe.population[0].raw_fitness = 1.0;
        
        let reproducers = spe.get_reproducers();

        assert_eq!(reproducers, vec![1, 2]);
    }

    // this test is a bit weak. should compare different length genomes
    #[test]
    fn test_alternating() {

        // We'll start by making dense genomes and then replace the edges
        let mut gen_1 = Genome::new_dense(2, 2);
        let mut gen_2 = Genome::new_dense(2, 2);

        // set the weights to something we can track
        for g in 0..gen_1.edge_genes.len() {

            // pull out unchanging things from the extant edge
            let innov = gen_1.edge_genes[g].innov;
            let in_node = gen_1.edge_genes[g].source_innov;
            let out_node = gen_1.edge_genes[g].target_innov;

            gen_1.edge_genes[g] = Edge::new(innov, in_node, out_node, 1.0);
            gen_2.edge_genes[g] = Edge::new(innov, in_node, out_node, 2.0);
        }

        // same for the nodes and their biases
        for g in 0..gen_1.node_genes.len() {

            gen_1.node_genes[g].bias = 1.0;
            gen_2.node_genes[g].bias = 2.0;
        }
        
        // build the alternating genome
        let params = AlignmentParams { crossover_mode: CrossoverMode::Alternating };
        let alternating_genome = Alignment::crossover(gen_1, 
                                                              gen_2, 
                                                              &params);


        assert_eq!(alternating_genome.edge_genes[0].weight, 1.0);
        assert_eq!(alternating_genome.edge_genes[1].weight, 2.0);
        assert_eq!(alternating_genome.edge_genes[2].weight, 1.0);
        assert_eq!(alternating_genome.edge_genes[3].weight, 2.0);

        assert_eq!(alternating_genome.node_genes[0].bias, 1.0);
        assert_eq!(alternating_genome.node_genes[1].bias, 2.0);
        assert_eq!(alternating_genome.node_genes[2].bias, 1.0);
        assert_eq!(alternating_genome.node_genes[3].bias, 2.0)

    }

    #[test]
    fn test_calculate_raw_fitness() {

        fn n_genes(gen: &Genome) -> f64 {
            gen.edge_genes.len() as f64
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