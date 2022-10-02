use rand::prelude::*;

use crate::network::node::*;
use crate::network::edge::Edge;

pub struct Genome {
    pub edge_genes: Vec<Edge>,
    pub node_genes: Vec<Node>,
    next_innovation: usize,
    pub sensor_idx: Vec<usize>,
    pub output_idx: Vec<usize>,
}

impl Genome {
    // genome that translates to a dense two-layer network
    fn new_minimal_dense(&self, sensors: usize, outputs: usize) -> Genome {

        // rng for gene initialization
        let mut rng = rand::thread_rng();

        // we can start with the nodes
        let mut node_genes = Vec::with_capacity(sensors + outputs);
        let mut sensor_idx = Vec::with_capacity(sensors);
        let mut output_idx = Vec::with_capacity(outputs);


        for si in 0..sensors {
            node_genes.push(Node::new(NodeKind::Sensor, 0.0));
            sensor_idx.push(si);
        }

        for oi in 0..outputs {
            node_genes.push(Node::new(NodeKind::Output, 
                                      rng.gen_range(-1.0..1.0)));
            output_idx.push(sensors + oi);
        }

        // iterate with each added edge gene
        let mut innovation = 0;
        let mut edge_genes: Vec<Edge> = Vec::with_capacity(sensors * outputs);
        
        // we will start with sensors fully connected to outputs
        for si in 0..sensors {
            for oi in 0..outputs {
                let unif_weight: f64 = rng.gen_range(-1.0..1.0);
                edge_genes.push(Edge::new(innovation, si, sensors + oi, unif_weight));
                innovation += 1;
            }
        }

        Genome {
            edge_genes,
            node_genes,
            next_innovation: innovation,
            sensor_idx,
            output_idx,
        }
    }
}