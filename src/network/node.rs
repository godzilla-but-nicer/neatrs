use crate::network::edge::Edge;

pub struct Node {
    active: bool,  // whether the node is active
    in_edges: Vec<&Edge>,
    out_edges: Vec<&Edge>,
}