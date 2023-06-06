#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub struct Node {
    pub innov: usize,                // innovation number for the nodes
    pub in_edges: Vec<usize>,        // incoming edges
    pub out_edges: Vec<usize>,       // we're going to want these for graph theory purposes
    pub output: f64,                 // signal to send to downstream nodes
    pub activation: fn(f64) -> f64,  // net_input -> output
    pub bias: f64,                   // bias that is added to node input
}

enum NodeKind {
    Input,
    Output,
    General,
}

impl Node {
    
    // edge operations
    // delete an edge from in edges
    pub fn remove_edge(&mut self, edge: usize) {
        self.in_edges.retain(|x| x != &edge);
    }

    // inverse of above
    pub fn add_edge(&mut self, edge: usize) {
        self.in_edges.push(edge);
    }

    // default constructor. edge logic handled in network
    pub fn new(innov: usize, bias: f64, activation: fn(f64) -> f64) -> Node {
        Node {
            in_edges: Vec::new(),
            out_edges: Vec::new(),
            output: 0.0,
            innov,
            bias,
            activation,
        }
    }

    pub fn from_edges(innov: usize, in_edges: Vec<usize>, out_edges: Vec<usize>, bias: f64) -> Node {
        
        Node {
            in_edges: in_edges,
            out_edges: out_edges,
            output: 0.0,
            activation: Node::sigmoid,
            innov,
            bias,
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
        let mut my_node = Node::new(0, 0.0, |x| x.abs());
        my_node.in_edges.push(1);
        my_node.in_edges.push(2);
        my_node.in_edges.push(3);
        my_node.remove_edge(2);
        assert_eq!(my_node.in_edges[0], 1);
        assert_eq!(my_node.in_edges[1], 3);
    }
}