use rand::prelude::*;
use rand_distr::Normal;

use crate::neural_network::node::*;
use crate::neural_network::edge::Edge;
use crate::community::community_params::GenomeParams;


#[derive(Clone)]
#[derive(PartialEq)]  // used in testing
#[derive(Debug)]
pub struct Genome {
    pub edge_genes: Vec<Edge>,
    pub node_genes: Vec<Node>,
    pub innovs: Vec<usize>,
    pub sensor_idx: Vec<usize>,  // indices of the sensor nodes
    pub output_idx: Vec<usize>,
    pub params: GenomeParams,
}

impl Genome {

    pub fn incompatibility(&self, partner: &Genome, max_innovation: usize) -> f64 {
        
        // align to count disjoint and excess genes
        let (aln_self, aln_partner) = self.align(partner, max_innovation);
        
        // get excess first
        let self_end = aln_self.iter().rposition(|x| *x != -1_i32).unwrap();
        let part_end = aln_self.iter().rposition(|x| *x != -1_i32).unwrap();
        let both_ends = vec![self_end, part_end];
        let excess = both_ends.iter().max().unwrap();

        // double counting some genes, corrected below
        let mut mismatches = 0;

        // we'll also count average weight difference here
        let mut diffs = 0.0;

        for i_num in 0..aln_self.len() {
            
            // neither has gene we just skip this gene
            if aln_self[i_num] == -1 && aln_self[i_num] == -1 {
                
            // note whether disjoint
            } else if aln_self[i_num] == -1 || aln_partner[i_num] == -1 {
                mismatches += 1

            // otherwise get the difference
            } else {
                let self_gi = self.edge_genes.iter().position(|gene| gene.innov == i_num).unwrap();
                let part_gi = partner.edge_genes.iter().position(|gene| gene.innov == i_num).unwrap();

                let self_weight = self.edge_genes[self_gi].weight;
                let part_weight = partner.edge_genes[part_gi].weight;

                diffs += (self_weight - part_weight).abs();
            }
        }
        // normalize
        let excess_norm = *excess as f64 / aln_self.len() as f64;
        let disjoint;
        if *excess < mismatches {
            disjoint = mismatches - *excess;
        } else {
            disjoint = mismatches;
        }
        let disjoint_norm = disjoint as f64 / aln_self.len() as f64;
        let avg_diff = diffs / aln_self.len() as f64;

        let incompatibility = self.params.disjoint_imp * disjoint_norm 
                            + self.params.excess_imp * excess_norm
                            + self.params.weight_imp * avg_diff;
        
        incompatibility
    }



    // produces vectors of innovation numbers in increasing order
    // and of the same length filled with -1 where disjoint or excess
    // right now we copy the edge_lists of both. could be better
    pub fn align(&self, partner: &Genome, total_innovations: usize) -> (Vec<i32>, Vec<i32>) {

        // alignment vectors of the two genomes
        let mut aln_self = Vec::new();
        let mut aln_partner = Vec::new();

        // iterate over possible genes the alignment
        for i_num in 0..total_innovations {

            // check if the gene_ is found in self
            let mut in_self = false;
            for s_gene in &self.edge_genes {
                if i_num == s_gene.innov {
                    in_self = true;
                    break
                }
            }

            let mut in_partner = false;            
            for p_gene in &partner.edge_genes {
                if i_num == p_gene.innov {
                    in_partner = true;
                    break
                }
            }

            // for each genome we add the innovation number if there
            // else we add a -1
            if in_self {
                aln_self.push(i_num as i32);
            } else {
                aln_self.push(-1);
            }

            // same for partner
            if in_partner {
                aln_partner.push(i_num as i32);
            } else {
                aln_partner.push(-1);
            }
        }

        // trim the end off of the innovation numbers
        let self_keep = aln_self.iter().rposition(|x| *x != -1_i32).unwrap() + 1;
        let partner_keep = aln_partner.iter().rposition(|x| *x != -1_i32).unwrap() + 1;

        if self_keep > partner_keep {
            aln_self.resize(self_keep, -1);
            aln_partner.resize(self_keep, -1);
        } else {
            aln_self.resize(partner_keep, -1);
            aln_partner.resize(partner_keep, -1);
        }


        (aln_self, aln_partner)

    }

    // mutation functions
    // we'll start simple. perturb all weights
    fn mutate_weights(&mut self) {
        let normal = Normal::new(0., self.params.weight_mut_sd).unwrap();
        for gene in &mut self.edge_genes {
            gene.weight += normal.sample(&mut rand::thread_rng());
        }
    }

    fn mutate_bias(&mut self) {
        let normal = Normal::new(0., self.params.bias_mut_sd).unwrap();
        for gene in &mut self.node_genes {
            gene.bias += normal.sample(&mut rand::thread_rng());
        }
    }

    // disable an edge gene
    fn enable_disable(&mut self, innov: usize) {
        let iidx = self.index_from_innov(innov);
        self.edge_genes[iidx].enabled = !self.edge_genes[iidx].enabled;
    }
    
    // add node "on" an existing edge
    fn insert_node(&mut self, innov: usize, max_innov: usize) {
        
        // identify ends of new edges
        let iidx = self.index_from_innov(innov);
        let old_source = self.edge_genes[iidx].source_i;
        let old_target = self.edge_genes[iidx].target_i;
        let node_i = self.node_genes.len();

        // construct new genes
        let mut rng = rand::thread_rng();
        let new_node = Node::new(NodeKind::Hidden, rng.gen_range(-1.0..1.0));
        let inner_edge = Edge::new(max_innov + 1, old_source, node_i, rng.gen_range(-1.0..1.0));
        let outer_edge = Edge::new(max_innov + 2, node_i, old_target, rng.gen_range(-1.0..1.0));

        // remove old edge and add new stuff
        self.node_genes.push(new_node);
        self.edge_genes.swap_remove(iidx);
        self.edge_genes.push(inner_edge);
        self.edge_genes.push(outer_edge);
    }

    fn add_connection(&mut self, source_i: usize, target_i: usize, max_innov: usize) {

        // construct the edge and add it to the list
        let mut rng = rand::thread_rng();
        let new_edge = Edge::new(max_innov + 1, source_i, target_i, rng.gen_range(-1.0..1.0));
        self.edge_genes.push(new_edge);
    }

    // used to construct tests
    pub fn _remove_by_innovation(&mut self, i_num: usize) {
        let gene_idx = self.edge_genes.iter()
                                    .position(|elem| elem.innov == i_num)
                                    .unwrap();
        self.edge_genes.remove(gene_idx);
    }

    // helper functions
    fn index_from_innov(&self, innov: usize) -> usize {

        let mut found = false;
        let mut iidx = 0;

        for gene_i in 0..self.edge_genes.len() {
            if self.edge_genes[gene_i].innov == innov {
                iidx = gene_i;
                found = true;
            }
        }

        if found {
            return iidx
        } else {
            panic!("Innovation number not found in genome");
        }
    }

    // genome that translates to a dense two-layer network
    pub fn new_dense(sensors: usize, outputs: usize) -> Genome {

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
        let mut innov_num = 0;
        let mut edge_genes: Vec<Edge> = Vec::with_capacity(sensors * outputs);
        let mut innovs = Vec::new();
        
        // we will start with sensors fully connected to outputs
        for si in 0..sensors {
            for oi in 0..outputs {
                let unif_weight: f64 = rng.gen_range(-1.0..1.0);
                edge_genes.push(Edge::new(innov_num, si, sensors + oi, unif_weight));
                innovs.push(innov_num);
                innov_num += 1;
            }
        }

        Genome {
            edge_genes,
            node_genes,
            innovs,
            sensor_idx,
            output_idx,
            params: GenomeParams::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incompatibility() {
        let mut gen_1 = Genome::new_dense(2, 3);
        let mut gen_2 = Genome::new_dense(3, 4);

        gen_1.params = GenomeParams::get_test_params();

        gen_2._remove_by_innovation(4);
        gen_2._remove_by_innovation(5);

        // knowns
        let disjoint = 2.;
        let excess = 6.;
        let N = 12.;

        let max_avg_weight_diff = 1.;

        let known = disjoint / N + excess / N;

        assert!((gen_1.incompatibility(&gen_2, 20) - known).abs() < max_avg_weight_diff)
    }

    #[test]
    fn test_align() {
        let mut gen_1 = Genome::new_dense(3, 4);
        let mut gen_2 = Genome::new_dense(3, 5);

        // remove a couple of genes
        gen_1._remove_by_innovation(3);
        gen_1._remove_by_innovation(4);
        gen_1._remove_by_innovation(6);
        gen_2._remove_by_innovation(6);
        gen_2._remove_by_innovation(11);

        let known_1 = vec![0, 1, 2, -1, -1, 5, -1, 7, 8, 9, 10, 11, -1, -1, -1];
        let known_2 = vec![0, 1, 2,  3,  4, 5, -1, 7, 8, 9, 10, -1, 12, 13, 14];

        let (aln_1, aln_2) = gen_1.align(&gen_2, 20);

        // debug
        println!("{:?}", aln_1);
        println!("{:?}", aln_2);

        assert_eq!(known_1, aln_1);
        assert_eq!(known_2, aln_2)
    }

    #[test]
    fn test_insert_node() {
        let mut gen = Genome::new_dense(3, 4);
        // mut innov = 5
        gen.insert_node(5, 11);
        assert!(gen.node_genes.len() == 8);
        assert!(gen.edge_genes.len() == 13);
        assert!(gen.edge_genes[11].source_i == 1);
        assert!(gen.edge_genes[11].target_i == 7);
        assert!(gen.edge_genes[12].source_i == 7);
        assert!(gen.edge_genes[12].target_i == 4);
    }

    #[test]
    fn test_add_connection() {
        let mut gen = Genome::new_dense(3, 4);
        gen._remove_by_innovation(5);
        gen.add_connection(1, 4, 11);

        let new_edge = gen.edge_genes.last().unwrap().clone();

        assert!(new_edge.innov == 12);
        assert!(new_edge.source_i == 1);
        assert!(new_edge.target_i == 4);
    }
}