use std::collections::{HashMap, HashSet, VecDeque};
use std::error;
use std::fmt;

use rstest::{fixture, rstest};
use topo_sort::{self, CycleError};

/// ConnectivityError is raised when outputs are not reachable from any sensors
#[derive(Clone, Copy, fmt::Debug, PartialEq)]
pub struct ConnectivityError;

impl fmt::Display for ConnectivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl error::Error for ConnectivityError {}

/// PathingError is raised when no path is available between two particular nodes
#[derive(Clone, Copy, fmt::Debug, PartialEq)]
pub struct PathingError;

impl fmt::Display for PathingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "No path available between nodes!")
    }
}

impl error::Error for PathingError {}

/// The Graph struct contains structural information about a network without the details needed to
/// fully compute activation. We use a Graph primarily to determine the activation order of nodes
/// in our neural network. It can also be used to compute other general graph structural properties.
/// In the current implementation sensors and outputs are owned by this struct which works fine.
/// However, this leads to extra copying elsewhere and I only only made this choice to get my tests
/// running. In the actual code, lifetime annotations ought to be sufficient to ensure valid
/// references to sensors and outputs from the neural network information.
pub struct Graph {
    edge_list: Vec<[usize; 2]>,
    preds: HashMap<usize, HashSet<usize>>,
    succs: HashMap<usize, HashSet<usize>>,
    sensors: Vec<usize>,
    outputs: Vec<usize>,
}

impl Graph {

    /// Default constructor for Graph that produces an empty graph with only a list of sensor 
    /// and output innovation numbers.
    pub fn new(sensors: Vec<usize>, outputs: Vec<usize>) -> Graph {
        Graph {
            edge_list: Vec::new(),
            preds: HashMap::new(),
            succs: HashMap::new(),
            sensors: sensors,
            outputs: outputs,
        }
    }

    /// This is a more useful constructor for graph that takes an edge list and produces a fully
    /// populated Graph object.
    /// 
    /// # Arguments
    /// 
    /// * `edge_list` - A Vector of arrays of innovation numbers, each representing a directed
    /// connection between nodes
    /// 
    /// * `sensors` - Reference to a vector of innovation numbers associated with the input nodes
    /// 
    /// * `outputs` - Reference to a vector of innovation numbers associated with the output nodes
    /// 
    /// # Example
    /// 
    /// ```
    /// let edges = vec![[0, 2], [0, 1], [1, 2]];
    /// let my_graph = Graph::from_edge_list(edges, &vec![0], &vec![2]);
    /// ```
    pub fn from_edge_list(
        edge_list: Vec<[usize; 2]>,
        sensors: Vec<usize>,
        outputs: Vec<usize>,
    ) -> Graph {
        let preds = Self::dependency_map(&edge_list, true);
        let succs = Self::dependency_map(&edge_list, false);

        Graph {
            edge_list,
            preds,
            succs,
            sensors,
            outputs,
        }
    }

    /// Produces a Hash lookup for neighbors of a node looking either forward or backward.
    /// 
    /// # Arguments
    /// 
    /// * `edge_list` - vector of arrays containing [source, target] innovation numbers of connected nodes
    /// * `backward` - determines whether we look forward for neighbors (successors) or backward for neighbors (predecessors)
    /// 
    /// # Example
    /// 
    /// ```
    /// let edges = vec![[0, 1], [1, 2], [0, 2]];
    /// let preds = Graph::dependency_map(&edges, True);
    /// assert_eq!(preds[2], vec![0, 1])
    /// ```
    pub fn dependency_map(
        edge_list: &Vec<[usize; 2]>,
        backward: bool,
    ) -> HashMap<usize, HashSet<usize>> {
        let mut dep_map: HashMap<usize, HashSet<usize>> = HashMap::new();

        for edge in edge_list {
            let key_node: usize;
            let val_node: usize;

            match backward {
                true => {
                    key_node = edge[1];
                    val_node = edge[0];
                }
                false => {
                    key_node = edge[0];
                    val_node = edge[1];
                }
            };

            // add key if its not there already
            dep_map.entry(key_node)
                   .or_default()  // if key is not present add the default empty HashSet
                   .insert(val_node);  // if it exists add the value to the HashSet
        }

        dep_map
    }

    /// Calculates an ordering of node innovation numbers according to the topological sort of the
    /// graph. If the graph is not a DAG this function throws CycleError.
    /// 
    ///  # Example
    /// 
    /// ```
    /// let edges = vec![[0, 1], [1, 2], [0, 2]];
    /// let g = Graph::from_edge_list(edges, 0, 2);
    /// assert_eq!(g.topological_sort().unwrap(), vec![0, 1, 2])
    /// ```
    pub fn topological_sort(&self) -> Result<Vec<usize>, CycleError> {
        let mut ts = topo_sort::TopoSort::with_capacity(self.preds.len());

        for (node, preds) in &self.preds {
            ts.insert(node, preds);
        }

        let mut nodes = Vec::new();
        
        for node in ts {

            match node {
                Ok((node, _)) => nodes.push(*node),
                Err(_) => return Err(topo_sort::CycleError),
            }
        }

        Ok(nodes)
    }

    /// Determines the activation order for an arbitrary neural network. This will either return a
    /// a valid sorting withour reference to sensors and outputs for DAG or return an invalid but
    /// useable ordering from the perspective of connecting sensors to outputs if cycles are 
    /// present. If the sensors are disconnected from the outputs this function throws a
    /// ConnectivityError.
    pub fn recurrent_pseudosort(&self) -> Result<Vec<usize>, ConnectivityError> {
        // essentially we will be flattening the layer map we get below
        let layer_map = match self.get_layer_map() {
            Ok(x) => x,
            Err(e) => return Err(e),
        };

        let mut layer_vec: Vec<Vec<usize>> = vec![vec![]];

        // we need a vector of vectors with one internal vector for each layer in the layer map.
        // we'll iterate over the map checking that the node can be placed into the vector resizing
        // the vector as needed (slow probably)
        let mut final_layer: usize = 0;
        for &node in layer_map.keys() {

            // check to make sure the node can be placed
            if layer_map[&node] > layer_vec.len() {
                // iteratively add inner vectors until we can accomodate the new node
                while layer_vec.len() - 1 < layer_map[&node] {
                    layer_vec.push(vec![]);
                }
            }

            // add the node to the inner vector corresponding to the mapped layer
            layer_vec[layer_map[&node]].push(node);
        }

        // flatten into an ordering
        let mut partial_order = Vec::new();

        for layer in &mut layer_vec {
            partial_order.append(layer)
        }

        Ok(partial_order.to_owned())

    }

    /// This function returns the shortest path length from any sensor to the target node.
    pub fn get_node_depth(&self, target_node: &usize) -> Result<usize, ConnectivityError> {
        let mut lengths: Vec<usize> = Vec::with_capacity(self.sensors.len());
        let mut num_errors = 0;  // ensures that we error out if all sensors fail to connect
        let mut depth: usize = 20;  // not sure why this is 20. i guess just arbitrary big number?

        // get all the path lengths and count unreachable inputs
        for sensor in &self.sensors {
            match self.get_path_length(sensor, target_node, true) {
                Some(x) => {
                    lengths.push(x);
                }
                None => {
                    num_errors += 1;
                }
            }
            if num_errors >= self.sensors.len() {
                return Err(ConnectivityError);
            }
        }

        for length in &lengths {
            if length < &depth {
                depth = *length;
            }
        }

        Ok(depth)
    }

    /// get shortest path between two nodes following edges either forward or backward.
    pub fn get_path_length(
        &self,
        source_node: &usize,
        target_node: &usize,
        backward: bool,
    ) -> Option<usize> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut steps: usize = 0;

        // convienent edge information
        let neighbors;
        if backward {
            neighbors = &self.preds;
        } else {
            neighbors = &self.succs;
        }

        // bfs for path length, not sure if we can do better
        queue.push_back(source_node);
        while queue.len() > 0 {
            let curr = queue.pop_front().unwrap();
            let mut next = VecDeque::new();

            // this is what we are searching for
            if curr == target_node {
                return Some(steps);
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
                queue = next.clone();
                next.clear();
                steps += 1;
            }
        }

        None
    }

    /// identify layers that can safely be activated in parallel starting from the sensor layer.
    /// Each key of the returned map is the innovation number of a node and its corresponding value
    /// is the layer to which is belongs. This function should maybe be broken up.
    pub fn get_layer_map(&self) -> Result<HashMap<usize, usize>, ConnectivityError> {
        // holds things we need to keep track of in bfs. visited if for recurrance
        let mut rev_layers: HashMap<usize, usize> = HashMap::with_capacity(self.preds.len());
        let mut layer = 0;
        let mut queue = VecDeque::new();

        // return error if we dig too deep without resolving ordering
        let max_depth = 50;

        // load the queue and the layer map with the outputs
        // we're going to do bfs backward, I don't quite remember why
        for output in &self.outputs {
            queue.push_back(output);
        }

        'bfs: while queue.len() > 0 {
            let curr = queue.pop_front().unwrap();
            let mut next_queue = VecDeque::new();


            // here we're doing some checks to seperate nodes with shared output to diff layers
            for node in &queue {

                // first if two nodes are fully connected we put the one closer to inputs to next
                if (self.preds[&curr].contains(&node)) & (self.succs[&curr].contains(&node)) {
                    let curr_depth = match self.get_node_depth(&curr) {
                        Ok(x) => x,
                        Err(e) => panic!("Disconnected node! {}", e),
                    };

                    let node_depth = match self.get_node_depth(&curr) {
                        Ok(x) => x,
                        Err(e) => panic!("Disconnected node! {}", e),
                    };

                    // push a node back if needed. if equal depth we leave them in place
                    if curr_depth < node_depth {
                        
                        next_queue.push_back(curr);
                        continue 'bfs;

                    } else if node_depth < curr_depth {
                        next_queue.push_back(*node);
                    }

                // if current node is needed for node in queue move curr to next
                } else if self.preds[node].contains(&curr) {
                    next_queue.push_back(curr);
                    continue 'bfs;

                // if node in queue is needed for current node move queue node to next queue
                } else if self.succs[node].contains(&curr) {
                    next_queue.push_back(*node);
                }
            }
            
            // remove any node that have been placed in the next queue from the current queue
            for shifted_node in &next_queue {

                let remove_index = match queue.iter().position(|x| x == shifted_node) {
                    Some(pos) => pos,
                    None => continue,
                };

                queue.remove(remove_index).unwrap();
            }

            // ok so our queue is safe for now we can move on through the search process
            rev_layers.insert(*curr, layer);

            // check if preds have been visited and add them to the next layers queue
            for in_node in &self.preds[&curr] {
                if rev_layers.contains_key(&in_node) {
                    next_queue.push_back(in_node);
                }
            }

            // move on to next layer
            if (queue.len() == 0) & (next_queue.len() > 0) {
                queue = next_queue;
                layer += 1;
            }

            // return error if max depth is exceeded
            if layer > max_depth {
                return Err(ConnectivityError);
            }
        }

        // now we have a map with the layers completely backward
        // with layer == the max depth
        
        // sensors potentially in different layers. lets push them all to the final reverse layer
        for sensor_node in &self.sensors {
            rev_layers.insert(*sensor_node, layer);
        }
        
        // now we can flip the reverse layers
        let mut node_layers = HashMap::new();
        for node in rev_layers.keys() {
            // reverse the layer ordering
            node_layers.insert(*node, layer - rev_layers[node]);
        }

        Ok(node_layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // test helper function
    #[fixture]
    fn singly_recurrant_edges() -> Vec<[usize; 2]> {
        vec![[0, 2], [1, 3], [3, 2], [3, 4], [2, 5], [4, 3], [4, 5]]
    }

    #[fixture]
    fn singly_recurrant_graph(singly_recurrant_edges: Vec<[usize; 2]>) -> Graph {
        Graph::from_edge_list(singly_recurrant_edges, vec![0, 1], vec![5])
    }

    #[fixture]
    fn three_cycle_graph() -> Graph {
        let edges = vec![
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 4],
            [4, 5],
            [5, 6],
            [5, 7],
            [5, 8],
            [6, 4],
            [6, 7],
            [6, 8],
        ];
        Graph::from_edge_list(edges, vec![0, 1], vec![7, 8])
    }

    #[fixture]
    fn two_cycle_graph() -> Graph {
        let edges = vec![[0, 2], [1, 2], [2, 3], [2, 4], [3, 2], [3, 4]];
        Graph::from_edge_list(edges, vec![0, 1], vec![4])
    }

    #[rstest]
    #[case(singly_recurrant_graph(singly_recurrant_edges()), 2, 1)]
    #[case(three_cycle_graph(), 5, 3)]
    #[case(two_cycle_graph(), 3, 2)]
    fn test_get_node_depth(#[case] graph: Graph, #[case] node: usize, #[case] expected: usize) {
        assert_eq!(graph.get_node_depth(&node).unwrap(), expected)
    }

    #[rstest]
    #[case(singly_recurrant_graph(singly_recurrant_edges()), 3, 5, 2)]
    #[case(three_cycle_graph(), 2, 7, 3)]
    #[case(two_cycle_graph(), 2, 4, 1)]
    fn test_path_length(
        #[case] graph: Graph,
        #[case] source: usize,
        #[case] target: usize,
        #[case] expected: usize,
    ) {
        assert_eq!(graph.get_path_length(&source, &target, true).unwrap(), expected)
    }

    #[rstest]
    fn test_get_layer_map(three_cycle_graph: Graph) {
        
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

        assert_eq!(three_cycle_graph.get_layer_map().unwrap(), known)
    }
}
