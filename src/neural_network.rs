pub mod edge;
pub mod node;

use crate::neural_network::node::Node;
use crate::neural_network::edge::Edge;
use crate::community::genome::{Genome, NodeGene};

struct NeuralNetwork {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    sensor_idx: Vec<usize>,
    output_idx: Vec<usize>,
    max_depth: usize
}

impl NeuralNetwork {

    // convert sensor values to outputs
    // because we cant guarentee the order of the nodes in the vec, we must
    // iterate until outputs are active. with each iteration the activation
    // depth grows by one layer.
    fn propagate(&mut self, inputs: Vec<f64>) {

        // iterated in outer loop. panics when > max_depth
        let mut layer_count: usize = 0;

        // activate the sensors
        self.load_sensors(inputs);

        // outer iteration which we continue until outputs are activated
        // is there an algorithm we could use here?
        while self.outputs_inactive() {

            // iteration over all nonsensor nodes to perform activations
            for node_i in self.sensor_idx.len()..self.nodes.len() {

                // never need to activate sensors
                if  ! self.sensor_idx.contains(&node_i) {

                    // if the node's inputs are active we activate
                    if self.node_ready(node_i) {
                        self.activate_node(node_i);
                    }
                }
            }

            // add 1 to the depth of activation
            layer_count += 1;
            if layer_count > self.max_depth {
                panic!("Max depth exceeded!!");
            }
        }
    }

    // check if a node is ready to be activated
    fn node_ready(&self, node_i: usize) -> bool {

        for edge_i in &self.nodes[node_i].in_edges {

            // locate the focal input
            let edge = &self.edges[*edge_i];
            let source_node = &self.nodes[edge.source_i];
            
            if source_node.active {
                return true
            }
        }
        // no input is active
        return false
    }

    // calculate net input and activate a node
    fn activate_node(&mut self, node_i: usize) {

        // init input with bias instead of adding later
        let mut net_input = self.nodes[node_i].bias;

        for edge_i in &self.nodes[node_i].in_edges {
            
            // locate the focal input
            let edge = &self.edges[*edge_i];
            let source = &self.nodes[edge.source_i];

            net_input += source.output * edge.weight;
        }
        self.nodes[node_i].output = (self.nodes[node_i].activation)(net_input);
        self.nodes[node_i].active = true
    }

    // checks whether all outputs are engaged
    fn outputs_inactive(&self) -> bool{
        for output_i in &self.output_idx {
            if !self.nodes[*output_i].active {
                return true
            }
        }
        return false
    }
    
    // return the values sent out from the output nodes
    // panics if called on inactive outputs
    fn get_outputs(&self) -> Vec<f64> {

        let mut output_vec = Vec::<f64>::with_capacity(self.output_idx.len());
        for output_i in &self.output_idx {

            if self.nodes[*output_i].active {
                output_vec.push(self.nodes[*output_i].output);
            } else {
                panic!("Outputs retrieved before activation!");
            }
        }
        output_vec
    }

    // activates the sensor nodes by setting the output to their observed value
    fn load_sensors(&mut self, values: Vec<f64>) {
        for (val_i, sensor_j) in self.sensor_idx.iter().enumerate() {
            self.nodes[*sensor_j].output = values[val_i];
            self.nodes[*sensor_j].active = true;
        }
    }
     

    // construct network from genome. primary constructor
    fn from_genome(genome: &Genome) -> NeuralNetwork {

        // make owned copies of objects
        let mut nodes = genome.node_genes.clone();
        let edges = genome.edge_genes.clone();

        // populate in edge indices for each node
        for node_i in 0..nodes.len() {
            for edge_i in 0..edges.len() {
                if node_i == edges[edge_i].target_i {
                    nodes[node_i].in_edges.push(edge_i);
                }
            }
        }

        NeuralNetwork {
            nodes,
            edges,
            sensor_idx: genome.sensor_idx.clone(),
            output_idx: genome.output_idx.clone(),
            max_depth: 20,
        }

    }
}
    
#[cfg(test)]
mod test_neural_network {
    use super::*;
    use crate::community::genome::Genome;

    #[test]
    fn test_from_genome() {
        // init the network from a genome
        let gen = Genome::new_dense(3, 4);
        let nn = NeuralNetwork::from_genome(&gen); 

        // debug
        for edge_i in 0..nn.edges.len() {
            println!("idx: {}, source: {}, weight: {}, target: {}", edge_i, 
                                                    nn.edges[edge_i].source_i,
                                                    nn.edges[edge_i].weight, 
                                                    nn.edges[edge_i].target_i);
                                                }
                                                
        for node_i in 0..nn.nodes.len() {
            println!("idx: {}, output: {}, kind: {:?}, in_edges: {:?}", node_i, 
                                                        nn.nodes[node_i].output,
                                                        nn.nodes[node_i].kind,
                                                        nn.nodes[node_i].in_edges)
        }

        assert!(nn.edges[5].source_i == 1);
        assert!(nn.nodes[6].in_edges[2] == 11)
    }

    #[test]
    fn test_node_ready() {
        let gen = Genome::new_dense(3, 2);
        let mut nn = NeuralNetwork::from_genome(&gen);
        nn.nodes[0].active = true;
        nn.nodes[1].active = true;
        nn.nodes[2].active = true;
        assert!(nn.node_ready(4))
    }
    
    #[test]
    fn test_propagate() {
        let gen = Genome::new_dense(3, 2);
        let mut nn = NeuralNetwork::from_genome(&gen);
        
        // set weights to something easy
        for i in 0..nn.edges.len() {
            nn.edges[i].weight = i as f64;
        }

        nn.nodes[3].bias = 2.0;
        nn.nodes[4].bias = 1.0;
        
        // same for the sensors
        nn.load_sensors(vec![-1., 0., 1.]);
        nn.propagate();
        
        // debug
        for edge in &nn.edges {
            println!("source: {}, weight: {}, target: {}", edge.source_i, edge.weight, edge.target_i);
        }

        for node_i in 0..nn.nodes.len() {
            println!("idx: {}, output: {}, kind: {:?}", node_i, nn.nodes[node_i].output, nn.nodes[node_i].kind)
        }

        // components of net input going into output
        let mut net_comps_3 = Vec::new();
        for in_edge_i in &nn.nodes[3].in_edges {
            let source_i = nn.edges[*in_edge_i].source_i;
            net_comps_3.push(nn.nodes[source_i].output * nn.edges[*in_edge_i].weight)
        }
        net_comps_3.push(nn.nodes[3].bias);

        let mut net_comps_4 = Vec::new();
        for in_edge_i in &nn.nodes[4].in_edges {
            let source_i = nn.edges[*in_edge_i].source_i;
            net_comps_4.push(nn.nodes[source_i].output * nn.edges[*in_edge_i].weight)
        }
        net_comps_4.push(nn.nodes[4].bias);
        println!("Node 3 inputs: {:?}", net_comps_3);
        println!("Node 4 inputs: {:?}", net_comps_4);
        
        let outputs = nn.get_outputs();
        
        assert!((outputs[0] - 0.997527).abs() < 1e-3);
        assert!((outputs[1] - 0.993307).abs() < 1e-3)
    }
}