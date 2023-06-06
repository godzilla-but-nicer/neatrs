pub mod edge;
pub mod node;

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::error;

use crate::community::genome::Genome;
use crate::neural_network::edge::Edge;
use crate::neural_network::node::Node;

use topo_sort::{self, CycleError};

#[derive(Clone, Copy, fmt::Debug, PartialEq)]
pub struct ConnectivityError;

impl fmt::Display for ConnectivityError {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl error::Error for ConnectivityError {}

#[derive(Clone, Copy, fmt::Debug, PartialEq)]
pub struct PathingError;

impl fmt::Display for PathingError {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "No path available between nodes!")
    }
}

impl error::Error for PathingError {}

struct NeuralNetwork {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    sensor_idx: Vec<usize>,
    output_idx: Vec<usize>,
    activation_order: Vec<usize>,
    node_map: HashMap<usize, usize>,  // this map innovation numbers to indices
}

impl NeuralNetwork {

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

        let mut nn = NeuralNetwork {
            nodes,
            edges,
            sensor_idx: genome.sensor_idx.clone(),
            output_idx: genome.output_idx.clone(),
            activation_order: Vec::new(),
            node_map: node_map
        };

        let ao = match nn.topological_sort_idx() {
            Ok(order) => order,
            Err(_) => nn.recurrent_pseudosort_idx(),
        };

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
    fn to_edge_list(&self) -> Vec<(usize, usize)> {
        let mut edges: Vec<(usize, usize)>;

        for edge in self.edges {
            edges.push((self.node_map[&edge.source_innov], self.node_map[&edge.target_innov]));
        }
        edges
    }

    // gets a vector of vectors for each node's dependency on other nodes
    fn to_dependency_map(&self, forward: bool) -> HashMap<usize, HashSet<usize>> {
        let mut dep_map: HashMap<usize, HashSet<usize>> = HashMap::new();

        for node_i in 0..self.nodes.len() {
            if forward == false {
                for in_edge_i in self.nodes[node_i].in_edges {
                    let dep_innov = self.edges[in_edge_i].source_innov;
                    dep_map[&node_i].insert(dep_innov);
                }
            } else {
                for out_edge_i in self.nodes[node_i].out_edges {
                    let dep_innov = self.edges[out_edge_i].target_innov;
                    dep_map[&node_i].insert(dep_innov);
                }
            }
        }

        return dep_map;
    }

    // assumes we have a DAG
    fn topological_sort_idx(&self) -> Result<Vec<usize>, CycleError> {
        let dep_list = self.to_dependency_map(true);

        let mut ts = topo_sort::TopoSort::with_capacity(dep_list.len());

        for dep_i in 0..dep_list.len() {
            ts.insert(dep_i, dep_list[&dep_i]);
        }

        match ts.into_vec_nodes() {
            topo_sort::SortResults::Full(sorted_nodes) => Ok(sorted_nodes),
            topo_sort::SortResults::Partial(_) => Err(topo_sort::CycleError),
        }
    }

    fn recurrent_pseudosort_idx(&self) -> Vec<usize> {

        // essentially we will be flattening the layer map we get below
        let layer_map = match self.get_layer_map() {
            Ok(x) => x,
            Err(e) => panic!("Error determining activation order: {}", e)
        };

        let mut layer_vec:Vec<Vec<usize>> = vec![vec![]];

        for node_i in layer_map.keys() {

            // we iteratively expand the vec if its not big enough
            if node_i > &(layer_vec.len() - 1) {
                for _ in (layer_vec.len() - 1)..*node_i {
                    layer_vec.push(vec![]);
                }
            }

            // add the node to the inner vector corresponding to the mapped layer
            layer_vec[layer_map[node_i]].push(*node_i);

        }

        let mut partial_order = Vec::new();

        for layer in &mut layer_vec {
            partial_order.append(layer)
        }

        partial_order
    }

    fn get_node_depth(&self, target_node: &usize) -> Result<usize, ConnectivityError> {

        let mut lengths: Vec<usize> = Vec::with_capacity(self.sensor_idx.len());
        let mut num_errors = 0;
        let mut depth: usize = 20;

        // get all the path lengths and count unreachable inputs
        for sensor in &self.sensor_idx {
            match self.path_length(sensor, target_node, true) {
                Some(x) => { lengths.push(x); },
                None => { num_errors += 1; },
            }
            if num_errors >= self.sensor_idx.len() {
                return Err(ConnectivityError)
            }
        }

        for length in &lengths {
            if length < &depth {
                depth = *length;
            }
        }

        Ok(depth)
    }
    
    // get shortest path between two nodes following edges either forward or backward
    fn path_length(&self, source_node: &usize, target_node: &usize, forward: bool) -> Option<usize> {

        let mut queue = VecDeque::new();
        let mut next = VecDeque::new();
        let mut visited = HashSet::new();
        let mut steps: usize = 0;

        // convienent edge information
        let neighbors;
        if forward {
            neighbors = self.to_dependency_map(true);
        } else {
            neighbors = self.to_dependency_map(false);
        }
        
        // bfs for path length, not sure if we can do better
        queue.push_back(source_node);
        while queue.len() > 0 {

            let curr = queue.pop_front().unwrap();

            // this is what we are searching for
            if curr == target_node {
                return Some(steps)
            }

            // don't want to get stuck in a loop!
            visited.insert(curr);

            for neighbor in &neighbors[curr] {
                if !visited.contains(neighbor) {
                    next.push_back(neighbor);
                }
            }

            // go to next step
            if (queue.len() == 0) & (next.len() > 0) {
                queue = next;
                next.clear();
                steps += 1;
            }
        }

        None
    }

    // identify layers that can safely be activated in parallel starting from the sensor layer
    fn get_layer_map(&self) -> Result<HashMap<usize, usize>, ConnectivityError> {
        // holds things we need to keep track of in bfs. visited if for recurrance
        let mut rev_layers: HashMap<usize, usize> = HashMap::with_capacity(self.nodes.len());
        let mut layer = 0;
        let mut queue = VecDeque::new();
        let mut next_queue = VecDeque::new();

        // return error if we dig too deep without resolving ordering
        let max_depth = 20;

        // load the queue and the layer map with the outputs
        // we're going to do bfs backward
        for output in self.output_idx {
            queue.push_back(output);
        }

        // objects that contain our topology information
        let ins = self.to_dependency_map(true);
        let outs = self.to_dependency_map(false);

        'bfs: while queue.len() > 0 {
            let curr = queue.pop_front().unwrap();

            // here we're doing some checks to seperate nodes with shared output to diff layers
            for node in &queue {
                // first if two nodes are fully connected we put the one closer to inputs to next
                if (ins[&curr].contains(&node)) & (outs[&curr].contains(&node)) {
                    let curr_depth =  match self.get_node_depth(&curr) {
                        Ok(x) => x,
                        Err(e) => panic!("Disconnected node! {}", e),
                    };

                    let node_depth =  match self.get_node_depth(&curr) {
                        Ok(x) => x,
                        Err(e) => panic!("Disconnected node! {}", e),
                    };

                    // push a node back if needed. if equal depth we leave them in place
                    if curr_depth > node_depth {

                        next_queue.push_back(curr);
                        continue 'bfs;

                    } else  if node_depth > curr_depth {
                        let remove_index = queue.iter().position(|x| x == node).unwrap();
                        queue.remove(remove_index).unwrap();
                        next_queue.push_back(*node);
                    }
                
                // if current node is needed for node in quueue move curr to next
                } else if ins[node].contains(&curr) {
                    next_queue.push_back(curr);
                    continue 'bfs;
                
                // if node in queue is needed for current node move queue node to next queue
                } else if outs[node].contains(&curr) {
                        let remove_index = queue.iter().position(|x| x == node).unwrap();
                        queue.remove(remove_index).unwrap();
                        next_queue.push_back(*node);
                }

            }

            // ok so our queue is safe for now we can move on through the search process
            rev_layers.insert(curr, layer);

            // check if preds have been visited and add them to the next layers queue
            for in_node in &ins[&curr] {
                if rev_layers.contains_key(&in_node) {
                    next_queue.push_back(*in_node);
                }
            }

            // move on to next layer
            if (queue.len() == 0) & (next_queue.len() > 0) {
                queue = next_queue;
                layer += 1;
            }

            // return error if max depth is exceeded
            if layer > max_depth {
                return Err(ConnectivityError)
            }
        }

        // now we have a map with the layers completely backward
        // with layer == the max depth
        // and with sensors potentially in different layers.
        let mut node_layers = HashMap::new();
        for node in rev_layers.keys() {

            // send all the sensors to one side
            if self.sensor_idx.contains(node) {
                rev_layers[node] = layer
            }

            // reverse the layer ordering
            node_layers.insert(*node, layer - rev_layers[node]);
        }


        Ok(node_layers)
    }

    fn from_edge_list(edge_list: Vec<(usize, usize)>) -> NeuralNetwork {

        let mut edges = Vec::new();
        let dummy_innov: usize = 0;

        let mut required_nodes = 0;

        // getting the edges is really easy
        for (source, target) in edge_list {
            
            // we assume that node indices start from zero and occupy all relevant integer values
            if required_nodes <= source {
                required_nodes = source + 1;
            }
            if required_nodes <= target {
                required_nodes = target + 1;
            }

            edges.push(Edge::new(dummy_innov, source, target, 1.0));
        }

        let mut nodes = Vec::new();
        let mut sensor_idx = Vec::new();
        let mut output_idx = Vec::new();

        // getting the nodes is a bit more involved
        let mut node_map: HashMap<usize, usize> = HashMap::new();
        for node_i in 0..required_nodes {

            node_map.insert(node_i, node_i);

            let mut next_node = Node::new(node_i, 0.0, |x| x);

            // we have to fill in the edge information
            for edge_i in 0..edges.len() {

                if edges[edge_i].source_innov == node_i {
                    next_node.out_edges.push(edge_i);
                }
                if edges[edge_i].target_innov == node_i {
                    next_node.in_edges.push(edge_i);
                }
            }

            // and also identify whether a node is a sensor or output
            if next_node.in_edges.len() == 0 {
                sensor_idx.push(node_i);
            }
            if next_node.out_edges.len() == 1 {
                output_idx.push(node_i);
            }
        }

        let mut nn = NeuralNetwork {
            nodes,
            edges,
            sensor_idx,
            output_idx,
            activation_order: vec![],
            node_map: node_map,
        };
        
        let ao = match nn.topological_sort_idx() {
            Ok(order) => order,
            Err(_) => nn.recurrent_pseudosort_idx(),
        };

        nn.activation_order = ao;

        nn

    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // test helper function
    fn singly_recurrant_topology() -> Vec<(usize, usize)> {
        vec![(0, 2),
             (1, 3),
             (3, 2),
             (3, 4),
             (2, 5),
             (4, 3),
             (4, 7)]
    }

    fn three_cycle_topology() -> Vec<(usize, usize)> {
        vec![(0, 2),
             (0, 3),
             (1, 2),
             (1, 3),
             (2, 4),
             (3, 4),
             (4, 5),
             (5, 6),
             (5, 7),
             (5, 8),
             (6, 4),
             (6, 7),
             (6, 8)]
    }

    fn two_cycle_topology() -> Vec<(usize, usize)> {
        vec![(0, 2),
             (1, 2),
             (2, 3),
             (2, 4),
             (3, 2),
             (3, 4)]
    }

    #[test]
    fn test_from_genome() {
        // init the network from a genome
        let gen = Genome::new_dense(3, 4);
        let nn = NeuralNetwork::from_genome(&gen);

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
        assert!(nn.nodes[6].in_edges[2] == 11)
    }


    #[test]
    fn test_propagate() {
        let mut nn = NeuralNetwork::from_edge_list(singly_recurrant_topology());

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

    #[test]
    fn test_from_edge_list() {
        let nn = NeuralNetwork::from_edge_list(singly_recurrant_topology());
        
        assert_eq!(nn.sensor_idx, vec![0, 1]);
        assert_eq!(nn.output_idx, vec![5]);
        assert!(nn.edges[2].source_innov == 3)
    }

    #[test]
    fn test_get_node_depth() {
        let nn = NeuralNetwork::from_edge_list(two_cycle_topology());

        assert_eq!(nn.get_node_depth(&3).unwrap(), 2)
    }

    #[test]
    fn test_path_length() {
        let nn = NeuralNetwork::from_edge_list(two_cycle_topology());

        assert_eq!(nn.path_length(&2, &4, true).unwrap(), 1)
    }

    #[test]
    fn test_get_layer_map() {
        let nn = NeuralNetwork::from_edge_list(three_cycle_topology());

        // known values
        let mut known: HashMap<usize, usize> = HashMap::new();
        known.insert(0, 0);
        known.insert(1, 0);
        known.insert(2, 1);
        known.insert(3, 1);
        known.insert(4, 2);
        known.insert(5, 3);
        known.insert(6, 4);
        known.insert(7, 5);
        known.insert(8, 5);

        assert_eq!(known, nn.get_layer_map().unwrap())
    }
}
