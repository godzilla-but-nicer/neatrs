use crate::network::node::{Node, NodeKind};

struct NeuralNetwork<'a> {
    nodes: Vec<Node>,
    max_depth: usize
}

impl <'a> NeuralNetwork<'a> {

    // convert sensor values to outputs
    // because we cant guarentee the order of the nodes in the vec, we must
    // iterate until outputs are active. with each iteration the activation
    // depth grows by one layer.
    fn propagate(&self) {

        // iterated in outer loop. panics when > max_depth
        let layer_count: usize = 0;

        // outer iteration which we continue until outputs are activated
        while self.outputs_inactive() {

            // iteration over all nodes to perform activations
            'all_nodes: for node in self.nodes {

                // reset the initial input
                let mut net_input = 0.0;

                // never need to activate sensors
                if !node.kind == NodeKind::Sensor {

                    // we can only activate if all predecessors are active
                    for in_edge in node.in_edges {
                         
                        // we can skip this node if inputs are inactive
                        if !in_edge.in_node.active {
                            continue 'all_nodes;
                            
                        // otherwise we add the input
                        } else {
                            net_input += in_edge.in_node.output * in_edge.weight;
                        }
                    }
                    // perform the actual activation
                    node.output = node.activation(net_input);
                    node.active = true;
                }  // end sensor if
            }  // end all_nodes loop
        }  // end outer loop
    }

    // checks whether all outputs are engage
    fn outputs_inactive(&self) -> bool{
        for node in self.nodes {
            if node.kind == NodeKind::Output && !node.active {
                return true
            }
        }
        false
    }
}