use crate::community::genome::Genome;
use crate::community::innovation_tracker::InnovationTracker;
use crate::neural_network::edge::Edge;
use crate::neural_network::node::Node;

use rand::prelude::*;
use rand_distr::Normal;

/// Returned object by main mutation function containing all of the information needed to update
/// genomes and to keep track of new innovations.
pub enum MutationInfo {
    Quantitative(Genome),
    Topological((Genome, NewInnovs)),
}

/// This struct is used to keep track of new innovations added by small structural mutations.
/// 
/// # Attributes
/// 
/// * `nodes_added` - Number of new node innovations
/// * `edges_added` - Number of new edge innovations
pub struct NewInnovs {
    pub nodes_added: usize,
    pub edges_added: usize,
}

/// The main mutation function that calls the other mutation functions. Whether a mutation is
/// structural or not is currently determined elsewhere and this function determines what kind of
/// structural mutation will occur or adjusts weights if the mutation is not structural
/// and returns a MutationInfo enum containing a new genome and information about new innovation
/// numbers where applicable.
pub fn mutate(genome: Genome, tracker: &InnovationTracker, structural: bool) -> MutationInfo {
    let mut rng = rand::thread_rng();

    // we do either a single random structural mutation or all weight mutations
    if structural {
        let roll: f64 = rng.gen();
        // first try adding a node
        if roll < genome.params.insert_prob {
            let mutate_i = (0..genome.edge_genes.len()).choose(&mut rng).unwrap();

            let new_genome = insert_node(&genome, &genome.edge_genes[mutate_i].innov, tracker);
            let innovations = NewInnovs {
                nodes_added: 1,
                edges_added: 2,
            };

            return MutationInfo::Topological((new_genome, innovations));

        // then try making a connection
        } else if roll < genome.params.connect_prob + genome.params.insert_prob {
            // we're going to restrict the topology such that
            // sensors have no in-degree and outputs have no out-degree
            let mut non_sensors = Vec::new();
            let mut non_outputs = Vec::new();

            for node_i in 0..genome.node_genes.len() {
                if !genome
                    .sensor_innovs
                    .contains(&genome.node_genes[node_i].innov)
                {
                    non_sensors.push(node_i);
                } else if !genome
                    .output_innovs
                    .contains(&genome.node_genes[node_i].innov)
                {
                    non_outputs.push(node_i);
                }
            }

            let source_i = *non_outputs.choose(&mut rng).unwrap();
            let target_i = *non_sensors.choose(&mut rng).unwrap();

            // finally we can make the connection
            let new_genome = add_connection(genome, source_i, target_i, tracker);
            let innovations = NewInnovs {
                nodes_added: 0,
                edges_added: 1,
            };

            return MutationInfo::Topological((new_genome, innovations));

        // otherwise toggle a random edge
        } else {
            let toggle_i = (0..genome.edge_genes.len()).choose(&mut rng).unwrap();
            let new_genome = enable_disable(&genome, &genome.edge_genes[toggle_i].innov);
            let innovations = NewInnovs {
                nodes_added: 0,
                edges_added: 0,
            };
            return MutationInfo::Topological((new_genome, innovations));
        }

    // if we're not doing a structural mutation we fiddle with weights
    } else {
        let updated_genome = mutate_weights(genome);
        let new_genome = mutate_bias(updated_genome);
        return MutationInfo::Quantitative(new_genome);
    }
}

/// Adjusts all edge weights in a genome by a standard normal ammount
fn mutate_weights(genome: Genome) -> Genome {
    let normal = Normal::new(0., genome.params.weight_mut_sd).unwrap();
    let mut new_genome = genome.clone();
    for gene in &mut new_genome.edge_genes {
        gene.weight += normal.sample(&mut rand::thread_rng());
    }

    new_genome
}

/// Adjusts all node biases by a standard normal ammount
fn mutate_bias(genome: Genome) -> Genome {
    let normal = Normal::new(0., genome.params.bias_mut_sd).unwrap();
    let mut new_genome = genome.clone();
    for gene in &mut new_genome.node_genes {
        gene.bias += normal.sample(&mut rand::thread_rng());
    }

    new_genome
}

/// Disable an active edge or enable an inactive edge in a reversable way
fn enable_disable(genome: &Genome, innov: &usize) -> Genome {
    let iidx = genome.edge_index_from_innov(*innov).unwrap();
    let mut new_genome = genome.clone();
    new_genome.edge_genes[iidx].enabled = !genome.edge_genes[iidx].enabled;

    new_genome
}

/// Adds a node to the network by splitting an edge in two with the new node sitting in the middle
fn insert_node(genome: &Genome, edge_innov: &usize, tracker: &InnovationTracker) -> Genome {
    // identify ends of new edges
    let iidx = genome.edge_index_from_innov(*edge_innov).unwrap();
    let old_source = genome.edge_genes[iidx].source_innov;
    let old_target = genome.edge_genes[iidx].target_innov;
    let mut new_genome = genome.clone();

    // construct new genes
    let mut rng = rand::thread_rng();
    let new_node_innov = tracker.node_max_innov + 1;
    let new_node = Node::new(new_node_innov, rng.gen_range(-1.0..1.0), |x| x);
    let inner_edge = Edge::new(
        tracker.edge_max_innov + 1,
        old_source,
        new_node_innov,
        rng.gen_range(-1.0..1.0),
    );

    let outer_edge = Edge::new(
        tracker.edge_max_innov + 2,
        new_node_innov,
        old_target,
        rng.gen_range(-1.0..1.0),
    );

    // remove old edge and add new stuff
    new_genome.node_genes.push(new_node);
    new_genome.edge_genes.swap_remove(iidx);
    new_genome.edge_genes.push(inner_edge);
    new_genome.edge_genes.push(outer_edge);

    new_genome
}

/// Adds a new active edge between two existing nodes
fn add_connection(
    genome: Genome,
    source_i: usize,
    target_i: usize,
    tracker: &InnovationTracker,
) -> Genome {
    let mut rng = rand::thread_rng();
    let mut new_genome = genome.clone();

    // create and add edge
    let new_edge = Edge::new(
        tracker.edge_max_innov + 1,
        source_i,
        target_i,
        rng.gen_range(-1.0..1.0),
    );
    new_genome.edge_genes.push(new_edge);

    new_genome
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[fixture]
    fn three_by_four() -> Genome {
        Genome::new_dense(3, 4)
    }

    #[fixture]
    fn three_by_four_tracker() -> InnovationTracker {
        InnovationTracker {
            node_max_innov: 6,
            edge_max_innov: 11,
        }
    }

    #[rstest]
    fn test_insert_node(three_by_four: Genome, three_by_four_tracker: InnovationTracker) {
        let mutated = insert_node(&three_by_four, &5, &three_by_four_tracker);
        assert!(mutated.node_genes.len() == 8);
        assert!(mutated.edge_genes.len() == 13);
        assert!(mutated.edge_genes[11].source_innov == 1);
        assert!(mutated.edge_genes[11].target_innov == 7);
        assert!(mutated.edge_genes[12].source_innov == 7);
        assert!(mutated.edge_genes[12].target_innov == 4);
    }

    #[rstest]
    fn test_add_connection(three_by_four: Genome, three_by_four_tracker: InnovationTracker) {
        let mut gen = three_by_four.clone();  // this needed to use fixture syntax
        gen._remove_by_innovation(5);
        let mutated = add_connection(gen, 1, 4, &three_by_four_tracker);

        let new_edge = mutated.edge_genes.last().unwrap().clone();

        assert!(new_edge.innov == 12);
        assert!(new_edge.source_innov == 1);
        assert!(new_edge.target_innov == 4);
    }
}
