use crate::network::node::{Node, NodeKind};
use crate::network::edge::Edge;
use crate::genome::genome::Genome;

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
    fn propagate(&mut self) {

        // iterated in outer loop. panics when > max_depth
        let mut layer_count: usize = 0;

        // outer iteration which we continue until outputs are activated
        while self.outputs_inactive() {

            // iteration over all nodes to perform activations
            for node_i in 0..self.nodes.len() {

                // never need to activate sensors
                if self.nodes[node_i].kind != NodeKind::Sensor {

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
            
            if !source_node.active {
                return false
            }
        }
        // no input is inactive
        return true
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
        let mut edges = genome.edge_genes.clone();

        // populate in edge indices for each node
        for node_i in 0..nodes.len() {
            for edge_i in 0..nodes.len() {
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

    // creates a dense network with the given number of inputs and outputs
    // used for tests
    fn new_minimal(sensors: usize, outputs: usize) -> NeuralNetwork {

        // first we add edges
        let mut edge_vec: Vec<Edge> = Vec::with_capacity(sensors * outputs);
        
        for si in 0..sensors {
            for oi in 0..outputs {
                edge_vec.push(Edge::new(0, si, sensors + oi, 0.0));
            }
        }
        
        // now we need to add all of the nodes
        let mut node_vec: Vec<Node> = Vec::with_capacity(sensors + outputs);

        // iterate ver the sensors to add them
        for si in 0..sensors {

            // iterate over outputs to specify edges
            let mut outward_edges: Vec<usize> = Vec::with_capacity(outputs);
            for oi in 0..outputs {
                outward_edges.push(si * outputs + oi);
            }

            // create the node
            let new_node = Node::from_edges(NodeKind::Sensor, 
                                            vec![]);

            node_vec.push(new_node);
        }

        // iterate over outputs to add the outputs
        for oi in 0..outputs {
            
            // iterate over sensors to specify edges
            let mut inward_edges: Vec<usize> = Vec::with_capacity(sensors);
            for si in 0..sensors {
                inward_edges.push(si * outputs + oi);
            }
            let new_node = Node::from_edges(NodeKind::Output, 
                                            inward_edges);
            node_vec.push(new_node);
        }

        // build and return
        NeuralNetwork {
            nodes: node_vec,
            edges: edge_vec,
            sensor_idx: (0..sensors).collect(),
            output_idx: (sensors..(sensors + outputs)).collect(),
            max_depth: 20,
        }
    }
}

#[cfg(test)]
mod test_neural_network {
    use super::*;

    #[test]
    fn test_new_minimal() {
        let nn = NeuralNetwork::new_minimal(3, 4);
        
        // debug
        for edge_i in 0..nn.edges.len() {
            println!("idx: {}, source: {}, weight: {}, target: {}", edge_i, 
                                                    nn.edges[edge_i].source_i,
                                                    nn.edges[edge_i].weight, 
                                                    nn.edges[edge_i].target_i);
        }

        for node_i in 0..nn.nodes.len() {
            println!("idx: {}, output: {}, kind: {:?}", node_i, 
                                                        nn.nodes[node_i].output,
                                                        nn.nodes[node_i].kind)
        }

        assert!(nn.edges[5].source_i == 1);
        assert!(nn.nodes[6].in_edges[0] == 3)
    }

    #[test]
    fn test_node_ready() {
        let mut nn = NeuralNetwork::new_minimal(3, 2);
        nn.nodes[0].active = true;
        nn.nodes[1].active = true;
        nn.nodes[2].active = true;
        assert!(nn.node_ready(4))
    }

    #[test]
    fn test_propagate() {
        let mut nn = NeuralNetwork::new_minimal(3, 2);

        // set weights to something easy
        for i in 0..nn.edges.len() {
            nn.edges[i].weight = i as f64;
        }
        
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


        let outputs = nn.get_outputs();
        println!("{}, {}", outputs[0], outputs[1]);

        assert!((outputs[0] - 4.).abs() < 1e-8);
        assert!((outputs[1] - 4.).abs() < 1e-8)
    }
}