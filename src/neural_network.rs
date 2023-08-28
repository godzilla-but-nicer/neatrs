pub mod edge;
pub mod node;
mod graph;

use crate::community::genome::Genome;
use crate::neural_network::edge::Edge;
use crate::neural_network::node::Node;
use crate::neural_network::graph::Graph;

use std::collections::HashMap;


struct NeuralNetwork<'a> {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    sensor_idx: Vec<usize>,
    output_idx: Vec<usize>,
    activation_order: Vec<usize>,
    topology: Graph<'a>,
    node_map: HashMap<usize, usize>,  // this map innovation numbers to indices
}

impl <'a> NeuralNetwork<'a> {

    // convert sensor values to outputs
    // all we have to do is iterate over the activation order and activate each node
    fn propagate(&mut self, input: Vec<f64>) -> Vec<f64> {

        self.load_sensors(input);

        for node_i in self.activation_order {
            self.activate_node(node_i)
        }

        self.get_output()
    }

    
    // calculate net input and activate a node
    fn activate_node(&mut self, node_i: usize) {
        // init input with bias instead of adding later
        let mut net_input = self.nodes[node_i].bias;

        for edge_i in &self.nodes[node_i].in_edges {
            // locate the focal input
            let edge = &self.edges[*edge_i];
            let source_i = self.node_map[&edge.source_innov];
            let source = &self.nodes[source_i];

            net_input += source.output * edge.weight;
        }
        self.nodes[node_i].output = (self.nodes[node_i].activation)(net_input);
    }


    // activates the sensor nodes by setting the output to their observed value
    fn load_sensors(&mut self, values: Vec<f64>) {
        for (val_i, sensor_j) in self.sensor_idx.iter().enumerate() {
            self.nodes[*sensor_j].output = values[val_i];
        }
    }

    // construct network from genome. primary constructor
    fn from_genome(genome: &Genome) -> NeuralNetwork {

        // make owned copies of objects
        let mut nodes = genome.node_genes.clone();
        let mut node_map: HashMap<usize, usize> = HashMap::new();

        let edges = genome.edge_genes.clone();


        // populate in edge indices for each node along with the node map
        for node_i in 0..nodes.len() {

            node_map.insert(nodes[node_i].innov, node_i);
            
            for edge_i in 0..edges.len() {
                if nodes[node_i].innov == edges[edge_i].target_innov {
                    nodes[node_i].in_edges.push(edge_i);
                }

                if nodes[node_i].innov == edges[edge_i].source_innov {
                    nodes[node_i].out_edges.push(edge_i)
                }
            }
        }

        let mut sensor_idx = Vec::new();        
        for sinnov in genome.sensor_innovs {
            let sidx = genome.node_index_from_innov(sinnov).unwrap();
            sensor_idx.push(sidx);
        }

        let mut output_idx = Vec::new();
        for oinnov in genome.output_innovs {
            let oidx = genome.node_index_from_innov(oinnov);
            output_idx.push(oinnov);
        }

        let mut nn = NeuralNetwork {
            nodes,
            edges,
            sensor_idx: genome.sensor_innovs.clone(),
            output_idx: genome.output_innovs.clone(),
            activation_order: Vec::new(),
            topology: Graph::new(&sensor_idx, &output_idx),
            node_map: node_map
        };

        let edge_list = nn.to_edge_list();
        let topology = Graph::from_edge_list(edge_list, &nn.sensor_idx, &nn.output_idx);

        let ao = match topology.topological_sort_idx() {
            Ok(order) => order,
            Err(_) => match topology.recurrent_pseudosort_idx() {
                Ok(order) => order,
                Err(e) => panic!("Failed to determine activation order! {}", e),
            },
        };

        nn.topology = topology;
        nn.activation_order = ao;

        nn
    }

    // read the values of the output nodes only
    fn get_output(&self) -> Vec<f64> {

        let mut output = Vec::with_capacity(self.output_idx.len());

        for output_i in self.output_idx {
            output.push(self.nodes[output_i].output);
        }

        output.to_owned()
    }

    // returns an edge_list for the graph
    fn to_edge_list(&self) -> Vec<[usize; 2]> {
        let mut edges: Vec<[usize; 2]>;

        for edge in self.edges {
            edges.push([self.node_map[&edge.source_innov], self.node_map[&edge.target_innov]]);
        }
        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn singly_recurrant_genome() -> Genome {
        let mut new_nodes = Vec::new();
        let node_bias = 1.0;

        for node_i in 0..=5 {
            new_nodes.push(Node::new(node_i, node_i as f64, |x| x));
        }

        let mut new_edges = Vec::new();
        let edge_weight = 2.0;
        new_edges.push(Edge::new(0, 0, 2, edge_weight));
        new_edges.push(Edge::new(1, 1, 3, edge_weight));
        new_edges.push(Edge::new(2, 3, 2, edge_weight));
        new_edges.push(Edge::new(3, 3, 4, edge_weight));
        new_edges.push(Edge::new(4, 2, 5, edge_weight));
        new_edges.push(Edge::new(5, 4, 3, edge_weight));
        new_edges.push(Edge::new(6, 4, 5, edge_weight));

        let sensor_innovs = vec![0, 1];
        let output_innovs = vec![5];

        Genome::new(new_nodes, new_edges, sensor_innovs, output_innovs)
    }

    #[rstest]
    fn test_from_genome(singly_recurrant_genome: Genome) {
        // init the network from a genome
        let nn = NeuralNetwork::from_genome(&singly_recurrant_genome);

        // debug
        for edge_i in 0..nn.edges.len() {
            println!(
                "idx: {}, source: {}, weight: {}, target: {}",
                edge_i,
                nn.edges[edge_i].source_innov,
                nn.edges[edge_i].weight,
                nn.edges[edge_i].target_innov
            );
        }

        for node_i in 0..nn.nodes.len() {
            println!(
                "idx: {}, output: {}, in_edges: {:?}",
                node_i, nn.nodes[node_i].output, nn.nodes[node_i].in_edges
            )
        }

        assert!(nn.edges[5].source_innov == 1);
        assert!(nn.nodes[6].in_edges[2] == 11);
        assert!(nn.sensor_idx[1] == 1)
    }


    #[rstest]
    fn test_propagate(singly_recurrant_genome: Genome) {
        let mut nn = NeuralNetwork::from_genome(&singly_recurrant_genome);

        // set biass to something easy
        for i in 0..nn.edges.len() {
            nn.nodes[i].bias = i as f64;
        }

        // set a couple of edges to something different
        nn.edges[2].weight = 2.0;
        nn.edges[4].weight = 3.0;

        // same for the sensors
        let output = nn.propagate(vec![-1., 1.]);

        // debug
        for edge in &nn.edges {
            println!(
                "source: {}, weight: {}, target: {}",
                edge.source_innov, edge.weight, edge.target_innov
            );
        }

        for node_i in 0..nn.nodes.len() {
            println!(
                "idx: {}, output: {}, bias: {:?}",
                node_i, nn.nodes[node_i].output, nn.nodes[node_i].bias
            )
        }
        
        assert!((output[0] - 29.0).abs() < 1e-3)
    }
}
