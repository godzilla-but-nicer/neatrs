use std::collections::{HashMap, HashSet, VecDeque};
use std::error;
use std::fmt;

use rstest::{fixture, rstest};
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
// here, sensors, outputs are innovation numbers
pub struct Graph<'a> {
    edge_list: Vec<[usize; 2]>,
    preds: HashMap<usize, HashSet<usize>>,
    succs: HashMap<usize, HashSet<usize>>,
    sensors: &'a Vec<usize>,
    outputs: &'a Vec<usize>,
}

impl<'a> Graph<'a> {

    // return empty graph
    pub fn new(sensors: &Vec<usize>, outputs: &Vec<usize>) -> Graph<'a> {
        Graph {
            edge_list: Vec::new(),
            preds: HashMap::new(),
            succs: HashMap::new(),
            sensors: sensors,
            outputs: outputs,
        }
    }

    pub fn from_edge_list(
        edge_list: Vec<[usize; 2]>,
        sensors: &Vec<usize>,
        outputs: &Vec<usize>,
    ) -> Graph<'a> {
        let preds = Self::dependency_map(edge_list, true);
        let succs = Self::dependency_map(edge_list, false);

        Graph {
            edge_list,
            preds,
            succs,
            sensors,
            outputs,
        }
    }

    // gets a vector of vectors for each node's dependency on other nodes
    pub fn dependency_map(
        edge_list: Vec<[usize; 2]>,
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
            if !dep_map.contains_key(&key_node) {
                dep_map.insert(key_node, HashSet::new());
            }
            dep_map[&key_node].insert(val_node);
        }

        dep_map
    }

    // assumes we have a DAG
    pub fn topological_sort_idx(&self) -> Result<Vec<usize>, CycleError> {
        let mut ts = topo_sort::TopoSort::with_capacity(self.preds.len());

        for dep_i in 0..self.preds.len() {
            ts.insert(dep_i, self.preds[&dep_i]);
        }

        match ts.into_vec_nodes() {
            topo_sort::SortResults::Full(sorted_nodes) => Ok(sorted_nodes),
            topo_sort::SortResults::Partial(_) => Err(topo_sort::CycleError),
        }
    }

    pub fn recurrent_pseudosort_idx(&self) -> Result<Vec<usize>, ConnectivityError> {
        // essentially we will be flattening the layer map we get below
        let layer_map = match self.get_layer_map() {
            Ok(x) => x,
            Err(e) => return Err(e),
        };

        let mut layer_vec: Vec<Vec<usize>> = vec![vec![]];

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

        Ok(partial_order)

    }

    pub fn get_node_depth(&self, target_node: &usize) -> Result<usize, ConnectivityError> {
        let mut lengths: Vec<usize> = Vec::with_capacity(self.sensors.len());
        let mut num_errors = 0;
        let mut depth: usize = 20;

        // get all the path lengths and count unreachable inputs
        for sensor in self.sensors {
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

    // get shortest path between two nodes following edges either forward or backward
    pub fn get_path_length(
        &self,
        source_node: &usize,
        target_node: &usize,
        backward: bool,
    ) -> Option<usize> {
        let mut queue = VecDeque::new();
        let mut next = VecDeque::new();
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
                queue = next;
                next.clear();
                steps += 1;
            }
        }

        None
    }

    // identify layers that can safely be activated in parallel starting from the sensor layer
    pub fn get_layer_map(&self) -> Result<HashMap<usize, usize>, ConnectivityError> {
        // holds things we need to keep track of in bfs. visited if for recurrance
        let mut rev_layers: HashMap<usize, usize> = HashMap::with_capacity(self.preds.len());
        let mut layer = 0;
        let mut queue = VecDeque::new();
        let mut next_queue = VecDeque::new();

        // return error if we dig too deep without resolving ordering
        let max_depth = 50;

        // load the queue and the layer map with the outputs
        // we're going to do bfs backward
        for output in self.outputs {
            queue.push_back(output);
        }

        'bfs: while queue.len() > 0 {
            let curr = queue.pop_front().unwrap();

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
                    if curr_depth > node_depth {
                        next_queue.push_back(curr);
                        continue 'bfs;
                    } else if node_depth > curr_depth {
                        let remove_index = queue.iter().position(|x| x == node).unwrap();
                        queue.remove(remove_index).unwrap();
                        next_queue.push_back(*node);
                    }

                // if current node is needed for node in quueue move curr to next
                } else if self.preds[node].contains(&curr) {
                    next_queue.push_back(curr);
                    continue 'bfs;

                // if node in queue is needed for current node move queue node to next queue
                } else if self.succs[node].contains(&curr) {
                    let remove_index = queue.iter().position(|x| x == node).unwrap();
                    queue.remove(remove_index).unwrap();
                    next_queue.push_back(*node);
                }
            }

            // ok so our queue is safe for now we can move on through the search process
            rev_layers.insert(*curr, layer);

            // check if preds have been visited and add them to the next layers queue
            for in_node in &self.preds[curr] {
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
        // and with sensors potentially in different layers.
        let mut node_layers = HashMap::new();
        for node in rev_layers.keys() {
            // send all the sensors to one side
            if self.sensors.contains(node) {
                rev_layers[node] = layer
            }

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
    fn singly_recurrant_graph(singly_recurrant_edges: Vec<[usize; 2]>) -> Graph<'static> {
        Graph::from_edge_list(singly_recurrant_edges, &vec![0, 1], &vec![5])
    }

    #[fixture]
    fn three_cycle_graph() -> Graph<'static> {
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
        Graph::from_edge_list(edges, &vec![0, 1], &vec![7, 8])
    }

    #[fixture]
    fn two_cycle_graph() -> Graph<'static> {
        let edges = vec![[0, 2], [1, 2], [2, 3], [2, 4], [3, 2], [3, 4]];
        Graph::from_edge_list(edges, &vec![0, 1], &vec![4])
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
