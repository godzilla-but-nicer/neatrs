use crate::network::edge::Edge;

// could put values here for the output values and do fancy pattern matching
pub enum NodeKind {
    Sensor,
    Hidden,
    Output,
}

pub struct Node<'a> {
    pub active: bool,                 // whether the node has been activated
    pub in_edges: Vec<&'a Edge<'a>>,  // incoming edges
    out_edges: Vec<&'a Edge<'a>>,     // outgoing edges
    pub output: f64,                  // signal to send to downstream nodes
    activation: fn(f64) -> f64,       // net_input -> output
    pub kind: NodeKind,               // what kind of node it is
}

impl<'a> Node<'a> {
    // get the net input into a node and activate it
    fn activate(&mut self) {

        // sum up the weighted input
        let mut net_input = 0.0;
        for edge in &self.in_edges {
            net_input += edge.in_node.output * edge.weight;
        }
        self.active = true;
        self.output = (self.activation)(net_input);
    }

    // constructors
    // used for tests only
    fn _new(kind: NodeKind) -> Node<'a> {
        Node {
            active: false,
            in_edges: Vec::new(),
            out_edges: Vec::new(),
            output: 0.0,
            activation: |x| x,
            kind: kind,
        }
    }
}