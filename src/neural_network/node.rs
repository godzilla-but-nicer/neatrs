// could put values here for the output values and do fancy pattern matching
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum NodeKind {
    Sensor,
    Hidden,
    Output,
}

#[derive(Clone)]
#[derive(PartialEq)]
pub struct Node {
    pub active: bool,                // whether the node has been activated
    pub in_edges: Vec<usize>,        // incoming edges
    pub output: f64,                 // signal to send to downstream nodes
    pub activation: fn(f64) -> f64,  // net_input -> output
    pub kind: NodeKind,              // what kind of node it is
    pub bias: f64,
}

impl Node {
    
    // edge operations
    // delete an edge from either in or out_edges
    pub fn remove_edge(&mut self, edge: usize, outward: bool) {
        self.in_edges.retain(|x| x != &edge);
    }

    // inverse of above
    pub fn add_edge(&mut self, edge: usize, outward: bool) {
        self.in_edges.push(edge);
    }

    // default constructor. edge logic handled in network
    pub fn new(kind: NodeKind, bias: f64) -> Node {

        // assign activation based on node kind
        let activation = match kind {
            NodeKind::Sensor => |x| x,
            NodeKind::Hidden => Node::sigmoid,
            NodeKind::Output => Node::sigmoid,
        };

        let bias = match kind {
            NodeKind::Sensor => 0.0,
            NodeKind::Hidden => bias,
            NodeKind::Output => bias,
        };

        Node {
            active: false,
            in_edges: Vec::new(),
            output: 0.0,
            activation,
            kind,
            bias,
        }
    }

    pub fn from_edges(kind: NodeKind, in_edges: Vec<usize>) -> Node {
        
        // assign activation based on node kind
        let activation = match kind {
            NodeKind::Sensor => |x| x,
            NodeKind::Hidden => Node::sigmoid,
            NodeKind::Output => |x| x,
        };

        Node {
            active: false,
            in_edges: in_edges,
            output: 0.0,
            activation,
            kind,
            bias: 0.0,
        }

    }

    // activation functions
    // basic logistic function for hidden units
    fn sigmoid(x: f64) -> f64 {
        1. / (1. + (-x).exp())
    }

    // useful for testing
    fn _heaviside(x: f64) -> f64 {
        if x > 0.0 {
            return 1.0
        } else {
            return 0.0
        }
    }
}

#[cfg(test)]
mod test_node {
    use super::*;

    fn test_remove_edge() {
        let mut my_node = Node::new(NodeKind::Sensor, 0.0);
        my_node.in_edges.push(1);
        my_node.in_edges.push(2);
        my_node.in_edges.push(3);
        my_node.remove_edge(2, false);
        assert_eq!(my_node.in_edges[0], 1);
        assert_eq!(my_node.in_edges[1], 3);
    }
}