use crate::network::node::Node;

pub struct Edge {
    in_node: &Node,
    out_node: &Node,
}