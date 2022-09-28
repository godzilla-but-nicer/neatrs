use crate::network::node::Node;


pub struct Edge<'a> {
    pub in_node: &'a Node<'a>,
    pub out_node: &'a Node<'a>,
    pub weight: f64,
}

impl<'a> Edge<'a> {
    fn new<'b>(in_node: &'b Node, out_node: &'b Node, weight: f64) -> Edge<'b> {
        Edge {
            in_node,
            out_node,
            weight
        }
    }

    
}