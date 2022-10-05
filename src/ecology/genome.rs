use rand::prelude::*;

use crate::network::node::*;
use crate::network::edge::Edge;

#[derive(Clone)]
pub struct Genome {
    pub edge_genes: Vec<Edge>,
    pub node_genes: Vec<Node>,
    pub innovs: Vec<usize>,
    pub sensor_idx: Vec<usize>,
    pub output_idx: Vec<usize>,
}

impl Genome {

    pub fn incompatibility(&self, partner: &Genome, c1: f64, c2: f64, c3: f64, max_innovation: usize) -> f64 {
        
        // align to count disjoint and excess genes
        let (aln_self, aln_partner) = self.align(partner, max_innovation);
        
        // get excess first
        let self_end = aln_self.iter().rposition(|x| *x != -1_i32).unwrap();
        let part_end = aln_self.iter().rposition(|x| *x != -1_i32).unwrap();
        let both_ends = vec![self_end, part_end];
        let excess = both_ends.iter().max().unwrap();

        // double counting some genes, corrected below
        let mut disjoint = 0;

        // we'll also count average weight difference here
        let mut diffs = 0.0;

        for i_num in 0..aln_self.len() {
            
            // note whether disjoint
            if aln_self[i_num] == -1 || aln_partner[i_num] == -1 {
                disjoint += 1

            // if not get the difference
            } else {
                let self_gi = self.edge_genes.iter().position(|gene| gene.innovation == i_num).unwrap();
                let part_gi = partner.edge_genes.iter().position(|gene| gene.innovation == i_num).unwrap();

                let self_weight = self.edge_genes[self_gi].weight;
                let part_weight = partner.edge_genes[part_gi].weight;

                diffs += (self_weight - part_weight).abs();
            }
        }
        // normalize
        let excess_norm = *excess as f64 / aln_self.len() as f64;
        let disjoint_norm = (disjoint - *excess) as f64 / aln_self.len() as f64;
        let avg_diff = diffs / aln_self.len() as f64;

        c1 * disjoint_norm + c2 * excess_norm * c3 * avg_diff
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
                if i_num == s_gene.innovation {
                    in_self = true;
                    break
                }
            }

            let mut in_partner = false;            
            for p_gene in &partner.edge_genes {
                if i_num == p_gene.innovation {
                    in_partner = true;
                    break
                }
            }

            // if in neither we dont make space in the alignment
            if in_self || in_partner {

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

    // used to construct tests
    fn _remove_by_innovation(&mut self, i_num: usize) {
        let gene_idx = self.edge_genes.iter()
                                    .position(|elem| elem.innovation == i_num)
                                    .unwrap();
        self.edge_genes.remove(gene_idx);
    }

    // genome that translates to a dense two-layer network
    pub fn new_minimal_dense(sensors: usize, outputs: usize) -> Genome {

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
        }
    }
}

#[cfg(test)]
mod test_genome {
    use super::*;

    #[test]
    fn test_incompatibility() {
        let mut gen_1 = Genome::new_minimal_dense(2, 3);
        let mut gen_2 = Genome::new_minimal_dense(3, 4);

        gen_2._remove_by_innovation(4);
        gen_2._remove_by_innovation(5);

        // knowns
        let disjoint = 2.;
        let excess = 6.;
        let N = 12.;

        let max_avg_weight_diff = 2.;

        let known = disjoint / N + excess / N;

        assert!((gen_1.incompatibility(&gen_2, 1., 1., 1., 20) - known).abs() < max_avg_weight_diff)
    }

    #[test]
    fn test_align() {
        let mut gen_1 = Genome::new_minimal_dense(3, 4);
        let mut gen_2 = Genome::new_minimal_dense(3, 5);

        // remove a couple of genes
        gen_1._remove_by_innovation(3);
        gen_1._remove_by_innovation(4);
        gen_1._remove_by_innovation(6);
        gen_2._remove_by_innovation(6);
        gen_2._remove_by_innovation(11);

        let known_1 = vec![0, 1, 2, -1, -1, 5, 7, 8, 9, 10, 11, -1, -1, -1];
        let known_2 = vec![0, 1, 2,  3,  4, 5, 7, 8, 9, 10, -1, 12, 13, 14];

        let (aln_1, aln_2) = gen_1.align(&gen_2, 20);

        // debug
        println!("{:?}", aln_1);
        println!("{:?}", aln_2);

        assert_eq!(known_1, aln_1);
        assert_eq!(known_2, aln_2)
    }
}