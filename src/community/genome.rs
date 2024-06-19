use rand::prelude::*;

use crate::neural_network::edge::Edge;
use crate::neural_network::node::*;


#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub struct GenomeParams {
    pub insert_prob: f64,
    pub connect_prob: f64,
    pub weight_mut_sd: f64,
    pub bias_mut_sd: f64,
}

impl GenomeParams {
    pub fn new() -> GenomeParams {
        GenomeParams {
            insert_prob: 0.5,
            connect_prob: 0.5,
            weight_mut_sd: 1.0,
            bias_mut_sd: 1.0,
        }
    }
    
    pub fn get_test_params() -> GenomeParams {
        GenomeParams {
            insert_prob: 0.4,
            connect_prob: 0.3,
            weight_mut_sd: 1.0,
            bias_mut_sd: 1.0,
        }
    }
}

#[derive(Clone, PartialEq)] // used in testing
#[derive(Debug)]
pub struct Genome {
    pub edge_genes: Vec<Edge>,
    pub node_genes: Vec<Node>,
    pub sensor_innovs: Vec<usize>, // innovs of the sensor nodes
    pub output_innovs: Vec<usize>,
    pub params: GenomeParams,
}

impl Genome {

    // used to construct tests
    pub fn _remove_by_innovation(&mut self, i_num: usize) {
        let gene_idx = self
            .edge_genes
            .iter()
            .position(|elem| elem.innov == i_num)
            .unwrap();
        self.edge_genes.remove(gene_idx);
    }

    // helper functions
    pub fn edge_index_from_innov(&self, innov: usize) -> Option<usize> {
        let mut found = false;
        let mut iidx = 0;

        for gene_i in 0..self.edge_genes.len() {
            if self.edge_genes[gene_i].innov == innov {
                iidx = gene_i;
                found = true;
                break;
            }
        }

        if found {
            return Some(iidx);
        } else {
            return None;
        }
    }

    pub fn node_index_from_innov(&self, innov: usize) -> Option<usize> {
        let mut found = false;
        let mut iidx = 0;

        for gene_i in 0..self.node_genes.len() {
            if self.node_genes[gene_i].innov == innov {
                iidx = gene_i;
                found = true;
                break;
            }
        }

        if found {
            return Some(iidx);
        } else {
            return None;
        }
    }

    // basic constructor that takes a list of edges and the indices of fixed nodes
    pub fn new(
        nodes: Vec<Node>,
        edges: Vec<Edge>,
        sensors: Vec<usize>,
        outputs: Vec<usize>,
    ) -> Genome {

        Genome {
            node_genes: nodes,
            edge_genes: edges,
            sensor_innovs: sensors,
            output_innovs: outputs,
            params: GenomeParams::new(),
        }

    }

    // genome that translates to a dense two-layer network
    pub fn new_dense(sensors: usize, outputs: usize) -> Genome {
        
        // we can start with the nodes
        let mut node_genes = Vec::with_capacity(sensors + outputs);
        let mut sensor_innovs = Vec::with_capacity(sensors);
        let mut output_innovs = Vec::with_capacity(outputs);

        for si in 0..sensors {
            node_genes.push(Node::new(si, 0.0, Node::linear));
            sensor_innovs.push(si);
        }

        for oi in 0..outputs {
            node_genes.push(Node::new(sensors + oi, 0.0, Node::linear));
            output_innovs.push(sensors + oi);
        }

        // iterate with each added edge gene
        let mut innov_num = 0;
        let mut edge_genes: Vec<Edge> = Vec::with_capacity(sensors * outputs);
        let mut innovs = Vec::new();

        // we will start with sensors fully connected to outputs
        for si in 0..sensors {
            for oi in 0..outputs {
                edge_genes.push(Edge::new(innov_num, si, sensors + oi, 0.0));
                innovs.push(innov_num);
                innov_num += 1;
            }
        }

        Genome {
            edge_genes,
            node_genes,
            sensor_innovs,
            output_innovs,
            params: GenomeParams::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use super::*;

    #[rstest]
    #[case(3, 3)]
    #[case(6, 666)]
    fn test_edge_index_from_innov(#[case] innov: usize, #[case] index: usize) {
        let g = Genome::new_dense(2, 3);

        let found_index: usize;
        match g.edge_index_from_innov(innov) {
            None => found_index = 666,
            Some(idx) => found_index = idx,
        };

        assert_eq!(found_index, index)

    }
    
    #[rstest]
    #[case(3, 3)]
    #[case(5, 666)]
    fn test_node_index_from_innov(#[case] innov: usize, #[case] index: usize) {
        let g = Genome::new_dense(2, 3);

        let found_index: usize;
        match g.node_index_from_innov(innov) {
            None => found_index = 666,
            Some(idx) => found_index = idx,
        };

        assert_eq!(found_index, index)

    }
}