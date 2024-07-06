use std::collections::{HashMap, HashSet, VecDeque};
use std::error;
use std::fmt;

use rstest::{fixture, rstest};
use topo_sort::{self, CycleError};

enum LayerAdjustment {
    MoveCurrent,
    MovePartner,
    NoChanges,
}

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
/// fully compute activation.
///
/// We use a Graph primarily to determine the activation order of nodes
/// in our neural network. It can also be used to compute other general graph structural properties.
/// In the current implementation sensors and outputs are owned by this struct which works fine.
/// However, this leads to extra copying elsewhere and I only only made this choice to get my tests
/// running.
///
/// In the actual code, lifetime annotations ought to be sufficient to ensure valid
/// references to sensors and outputs from the neural network information.
#[derive(Clone)]
pub struct Graph {
    pub edge_list: Vec<[usize; 2]>,
    pub preds: HashMap<usize, HashSet<usize>>,
    pub succs: HashMap<usize, HashSet<usize>>,
    sensors: Vec<usize>,
    outputs: Vec<usize>,
}

impl Graph {
    /// Default constructor for Graph that produces an empty graph.
    pub fn new() -> Graph {
        Graph {
            edge_list: Vec::new(),
            preds: HashMap::new(),
            succs: HashMap::new(),
            sensors: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// This is a more useful constructor for graph that takes an edge list and produces a fully
    /// populated Graph object.
    ///
    /// # Example
    ///
    /// ```
    /// use neatrs::neural_network::graph::Graph;
    /// 
    /// let edges = vec![[0, 2], [0, 1], [1, 2]];
    /// let my_graph = Graph::from_edge_list(edges, vec![0], vec![2]);
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
    /// # Example
    ///
    /// ```
    /// use neatrs::neural_network::graph::Graph;
    /// use std::collections::HashSet;
    /// 
    /// let edges = vec![[0, 1], [1, 2], [0, 2]];
    /// let preds = Graph::dependency_map(&edges, true);
    /// 
    /// let mut known: HashSet<usize> = HashSet::new();
    /// known.insert(0);
    /// known.insert(1);
    /// 
    /// assert_eq!(preds[&2], known)
    /// ```
    pub fn dependency_map(
        edge_list: &Vec<[usize; 2]>,
        backward: bool,
    ) -> HashMap<usize, HashSet<usize>> {
        let mut dep_map: HashMap<usize, HashSet<usize>> = HashMap::new();

        for edge in edge_list {
            let key_node: usize;
            let val_node: usize;

            // want to ensure that all nodes are keys in the map regardless

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
            dep_map
                .entry(key_node)
                .or_default() // if key is not present add the default empty HashSet we're checking this twice now
                .insert(val_node); // if it exists add the value to the HashSet

            // also make sure the other node exists as key
            dep_map.entry(val_node).or_default();
        }

        dep_map
    }

    /// Calculates an ordering of node innovation numbers according to the topological sort of the
    /// graph. If the graph is not a DAG this function throws CycleError.
    ///
    ///  # Example
    ///
    /// ```
    /// use neatrs::neural_network::graph::Graph;
    /// 
    /// let edges = vec![[0, 1], [1, 2], [0, 2]];
    /// let g = Graph::from_edge_list(edges, vec![0], vec![2]);
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

    /// Determines the activation order for an arbitrary neural network.
    ///
    /// This will either return a
    /// a valid sorting withour reference to sensors and outputs for DAG or return an invalid but
    /// useable ordering from the perspective of connecting sensors to outputs if cycles are
    /// present.
    ///
    /// If the sensors are disconnected from the outputs this function throws a
    /// ConnectivityError.
    ///
    /// # Example
    ///
    /// ```
    /// use neatrs::neural_network::graph::Graph;
    /// 
    /// let edges = vec![[0, 1], [0, 2], [1, 2]];
    /// let g = Graph::from_edge_list(edges, vec![0], vec![2]);
    /// assert_eq!(g.recurrent_pseudosort(), Ok(vec![0, 1, 2]))
    /// ```
    pub fn recurrent_pseudosort(&self) -> Result<Vec<usize>, ConnectivityError> {
        // essentially we will be flattening the layer map we get below
        let layer_map = match self.get_layer_map() {
            Ok(x) => x,
            Err(e) => return Err(e),
        };

        let max_layer = layer_map.values().into_iter().max().unwrap();

        let mut layer_vec: Vec<Vec<usize>> = vec![vec![]; *max_layer + 1];

        // we need a vector of vectors with one internal vector for each layer in the layer map.
        // we'll iterate over the map checking that the node can be placed into the vector resizing
        // the vector as needed (slow probably)
        for &node in layer_map.keys() {
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
    ///
    /// # Example
    ///
    /// ```
    /// use neatrs::neural_network::graph::Graph;
    /// 
    /// let edges = vec![[0, 2], [1, 2], [1, 3], [2, 3]];
    /// let g = Graph::from_edge_list(edges, vec![0, 1], vec![3]);
    /// assert_eq!(g.get_node_depth(&3), Ok(1));
    /// ```
    pub fn get_node_depth(&self, target_node: &usize) -> Result<usize, ConnectivityError> {
        let mut lengths: Vec<usize> = Vec::with_capacity(self.sensors.len());
        let mut num_errors = 0; // ensures that we error out if all sensors fail to connect
        let mut depth: usize = 20; // not sure why this is 20. i guess just arbitrary big number?

        // get all the path lengths and count unreachable inputs
        for sensor in &self.sensors {
            match self.get_path_length(sensor, target_node, false) {
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
        // holds things we need to keep track of in bfs
        let mut rev_layers: HashMap<usize, usize> = HashMap::with_capacity(self.preds.len());
        let mut layer = 0;
        let mut queue = VecDeque::new();

        // return error if we dig too deep without resolving ordering. should be moved to variable
        let max_depth = 50;

        // load the queue and the layer map with the outputs
        // we're going to do bfs backward, I don't quite remember why
        for output in &self.outputs {
            queue.push_back(*output);
        }

        'bfs: while queue.len() > 0 {
            let curr = &queue.pop_front().unwrap();
            let mut next_queue = VecDeque::new();

            // here we're doing some checks to seperate nodes with shared output to diff layers
            for node in &queue {
                let node_preceeds_current = self.preds[curr].contains(node);
                let current_preceeds_node = self.preds[node].contains(curr);

                // first if two nodes are fully connected we put the one closer to inputs to next queue
                if node_preceeds_current & current_preceeds_node {
                    match self.resolve_fully_connected_pair(curr, node) {
                        LayerAdjustment::MoveCurrent => {
                            next_queue.push_back(*curr);
                            continue 'bfs;
                        }
                        LayerAdjustment::MovePartner => {
                            next_queue.push_back(*node);
                            continue;
                        }
                        LayerAdjustment::NoChanges => {}
                    }

                // if current node is needed for node in queue move curr to next
                } else if current_preceeds_node & !node_preceeds_current {
                    next_queue.push_back(*curr);
                    continue 'bfs;

                // if node in queue is needed for current node move queue node to next queue
                } else if node_preceeds_current & !current_preceeds_node {
                    next_queue.push_back(*node);
                }
            }

            // we now know the current node belongs here
            rev_layers.insert(*curr, layer);

            // add predecessors to next layer
            next_queue = self.finalize_next_queue(next_queue, &curr, &rev_layers);

            // remove any node that have been placed in the next queue from the current queue
            queue = Self::correct_queue(queue, &next_queue);

            // move on to next layer if ready
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
        // with layer == depth of network

        let node_layers = self.reverse_layer_map(rev_layers, layer);

        Ok(node_layers)
    }

    fn resolve_fully_connected_pair(
        &self,
        current_node: &usize,
        check_node: &usize,
    ) -> LayerAdjustment {
        let current_depth = match self.get_node_depth(current_node) {
            Ok(x) => x,
            Err(e) => panic!("Disconnected node! {}", e),
        };

        let check_depth = match self.get_node_depth(check_node) {
            Ok(x) => x,
            Err(e) => panic!("Disconnected node! {}", e),
        };

        // push a node back if needed. if equal depth we leave them in place
        if current_depth < check_depth {
            LayerAdjustment::MoveCurrent
        } else if check_depth < current_depth {
            LayerAdjustment::MovePartner
        } else {
            LayerAdjustment::NoChanges
        }
    }

    fn finalize_next_queue(
        &self,
        mut tentative_queue: VecDeque<usize>,
        current_node: &usize,
        reverse_layers: &HashMap<usize, usize>,
    ) -> VecDeque<usize> {
        let predecessors = self.preds.get(current_node).unwrap();
        for in_node in predecessors {
            if !reverse_layers.contains_key(in_node) {
                tentative_queue.push_back(*in_node);
            }
        }

        tentative_queue.clone()
    }

    fn correct_queue(mut current_queue: VecDeque<usize>, next_layer_queue: &VecDeque<usize>) -> VecDeque<usize> {

        for shifted_node in next_layer_queue {
            let remove_index = match current_queue.iter().position(|x| x == shifted_node) {
            Some(pos) => pos,
            None => continue,
            };
        
            current_queue.remove(remove_index).unwrap();
        }

    current_queue.clone()
        
    }

    fn reverse_layer_map(
        &self,
        mut reverse_layers: HashMap<usize, usize>,
        deepest_layer: usize,
    ) -> HashMap<usize, usize> {
        // sensors potentially in different layers or we might have pushed back internal nodes so
        // far that they are in a sensor layer. lets just push all of the sensors to a new final
        // reverse layer
        for sensor_node in &self.sensors {
            reverse_layers.insert(*sensor_node, deepest_layer + 1);
        }

        // now we can flip the reverse layers. We might be skipping layers in terms of the layer
        // labels but that shouldn't matter once the sorting is done
        let mut node_layers = HashMap::new();
        for node in reverse_layers.keys() {
            // reverse the layer ordering
            node_layers.insert(*node, deepest_layer + 1 - reverse_layers[node]);
        }

        node_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // test helper function
    #[fixture]
    fn singly_recurrent_edges() -> Vec<[usize; 2]> {
        vec![[0, 2], [1, 3], [3, 2], [3, 4], [2, 5], [4, 3], [4, 5]]
    }

    #[fixture]
    fn singly_recurrent_graph() -> Graph {
        Graph::from_edge_list(
            vec![[0, 2], [1, 3], [3, 2], [3, 4], [2, 5], [4, 3], [4, 5]],
            vec![0, 1],
            vec![5],
        )
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

    #[fixture]
    fn simplest_graph() -> Graph {
        Graph::from_edge_list(vec![[0, 2], [1, 2]], vec![0, 1], vec![2])
    }

    #[fixture]
    fn output_recurrance_graph() -> Graph {
        Graph::from_edge_list(vec![[0, 2], [1, 2], [2, 1]], vec![0, 1], vec![2])
    }

    #[fixture]
    fn singly_recurrent_layer_map_known() -> Vec<HashMap<usize, usize>> {
        let mut known1: HashMap<usize, usize> = HashMap::new();
        known1.insert(0, 0);
        known1.insert(1, 0);
        known1.insert(2, 2);
        known1.insert(3, 1);
        known1.insert(4, 2);
        known1.insert(5, 3);

        let mut known2: HashMap<usize, usize> = HashMap::new();
        known2.insert(0, 0);
        known2.insert(1, 0);
        known2.insert(2, 3);
        known2.insert(3, 2);
        known2.insert(4, 3);
        known2.insert(5, 4);


        vec![known1, known2]
    }
    
    #[fixture]
    fn two_cycle_layer_map_known() -> Vec<HashMap<usize, usize>> {
        let mut known1: HashMap<usize, usize> = HashMap::new();
        known1.insert(0, 0);
        known1.insert(1, 0);
        known1.insert(2, 1);
        known1.insert(3, 2);
        known1.insert(4, 3);

        let mut known2: HashMap<usize, usize> = HashMap::new();
        known2.insert(0, 0);
        known2.insert(1, 0);
        known2.insert(2, 2);
        known2.insert(3, 3);
        known2.insert(4, 4);
        
        vec![known1, known2]
    }

    #[fixture]
    fn three_cycle_layer_map_known() -> Vec<HashMap<usize, usize>> {

        let mut known1: HashMap<usize, usize> = HashMap::new();
        known1.insert(0, 0);
        known1.insert(1, 0);
        known1.insert(2, 1);
        known1.insert(3, 1);
        known1.insert(4, 2);
        known1.insert(5, 3);
        known1.insert(6, 4);
        known1.insert(7, 5);
        known1.insert(8, 5);
        
        let mut known2: HashMap<usize, usize> = HashMap::new();
        known2.insert(0, 0);
        known2.insert(1, 0);
        known2.insert(2, 2);
        known2.insert(3, 2);
        known2.insert(4, 3);
        known2.insert(5, 4);
        known2.insert(6, 5);
        known2.insert(7, 6);
        known2.insert(8, 6);

        vec![known1, known2]
    }

    #[rstest]
    #[case(singly_recurrent_graph(), 2, 1)]
    #[case(three_cycle_graph(), 5, 3)]
    #[case(two_cycle_graph(), 3, 2)]
    fn test_get_node_depth(#[case] graph: Graph, #[case] node: usize, #[case] expected: usize) {
        assert_eq!(graph.get_node_depth(&node).unwrap(), expected)
    }

    #[rstest]
    #[case(singly_recurrent_graph(), 3, 5, 2, false)]
    #[case(three_cycle_graph(), 2, 7, 3, false)]
    #[case(two_cycle_graph(), 4, 2, 1, true)]
    fn test_path_length(
        #[case] graph: Graph,
        #[case] source: usize,
        #[case] target: usize,
        #[case] expected: usize,
        #[case] backward_flag: bool,
    ) {
        assert_eq!(
            graph
                .get_path_length(&source, &target, backward_flag)
                .unwrap(),
            expected
        )
    }

    #[rstest]
    #[case(singly_recurrent_graph(), singly_recurrent_layer_map_known())]
    #[case(two_cycle_graph(), two_cycle_layer_map_known())]
    #[case(three_cycle_graph(), three_cycle_layer_map_known())]
    fn test_get_layer_map(#[case] graph: Graph, #[case] knowns: Vec<HashMap<usize, usize>>) {
        let mut solution_in_possible_knowns = false;
        let computed_layer_map = graph.get_layer_map().unwrap();

        println!("{:?}", computed_layer_map);

        for known in knowns {
            println!("known: {:?}", &known);
            if known == computed_layer_map {
                solution_in_possible_knowns = true;
            }
        }

        assert!(solution_in_possible_knowns)
    }

    #[rstest]
    #[case(singly_recurrent_graph(), vec![Ok(vec![0, 1, 3, 2, 4, 5]),
                                          Ok(vec![1, 0, 3, 2, 4, 5]),
                                          Ok(vec![0, 1, 3, 4, 2, 5]),
                                          Ok(vec![1, 0, 3, 4, 2, 5])])]
    #[case(two_cycle_graph(), vec![Ok(vec![0, 1, 2, 3, 4]), Ok(vec![1, 0, 2, 3, 4])])]
    #[case(simplest_graph(), vec![Ok(vec![0, 1, 2]), Ok(vec![1, 0, 2])])]
    #[case(output_recurrance_graph(), vec![Ok(vec![0, 1, 2]), Ok(vec![1, 0, 2])])]
    fn test_recurrent_pseudosort(
        #[case] graph: Graph,
        #[case] knowns: Vec<Result<Vec<usize>, ConnectivityError>>,
    ) {
        let mut solution_in_possible_knowns = false;
        let sort_result = graph.recurrent_pseudosort();

        for known in knowns {
            println!("a known: {:?}", known);
            if known == sort_result {
                solution_in_possible_knowns = true
            }
        }
        println!("{:?}", sort_result);
        assert!(solution_in_possible_knowns)
    }
}
