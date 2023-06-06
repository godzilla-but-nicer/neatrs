use crate::neural_network::edge::{Edge, EdgeGene};

pub enum EdgeLocus {
    Gene(EdgeGene),
    Gap,
}

pub enum NodeLocus {
    Gene(NodeGene),
    Gap,
}

#[derive(Debug)]
struct Alignment {
    // actual objects upon which recombination acts
    edgeomes: Vec<Vec<EdgeLocus>>,
    nodeomes: Vec<Vec<NodeGene>>,

    // max innovation numbers
    max_edge_innov: usize,
    max_node_innov: usize,
}

impl Alignment {
    pub fn from_parents(p1: Genome, p2: Genome) -> Alignment {
        // alignment vectors of the two genomes
        let mut p1_innovs = Vec::new();
        let mut p2_innovs = Vec::new();

        // pull out the innovation numbers and edges
        for p1_edge in &p1.edge_genes {
            p1_innovs.push(p1_edge.innov);
        }

        for p2_edge in &p2.edge_genes {
            p2_innovs.push(p2_edge.innov);
        }

        // fill any disjoint or excess edges with dummy edges
        let max_vals = vec![p1_innovs.max(), p2_innovs.max()];
        let max_edge_innov = max_vals.max();

        // these hold the loci of the genomes that will be aligned
        let p1_edgeome = order_edges(&p1, max_edge_innov);
        let p2_edgeome = order_edges(&p2, max_edge_innov);

        let edgeomes = vec![p1_edgeome, p2_edgeome];

        // Now we have to handle the nodes with the same proceedure
        let mut p1_node_innovs: Vec<usize> = Vec::new();
        let mut p2_node_innovs: Vec<usize> = Vec::new();

        // get the max innovation number again
        for p1_node in &p1.node_genes {
            p1_node_innovs.push(p1_node.innov)
        }

        for p2_node in &p2.node_genes {
            p2_node_innovs.push(p2_node.innov)
        }

        let max_node_vals = vec![p1_node_innovs.max(), p2_node_innovs.max()];
        let max_node_innov = max_node_vals.max();

        // get the aligned nodeomes
        let p1_nodeome = order_nodes(p1, max_node_innov);
        let p2_nodeome = order_nodes(p2, max_node_innov);

        let nodeomes = vec![p1_nodeome, p2_nodeome];

        Alignment {
            edgeomes,
            nodeomes,
            max_edge_innov,
            max_node_innov,
        }
    }

    // returns the edge genes of a parent in order and with gaps
    fn order_edges(parent: &Genome, innov_num: usize) -> Vec<EdgeLocus> {
        let par_edgeome: Vec<EdgeLocus> = Vec::new();

        for i in 0..=max_innov {
            par_gene_idx = parent.index_from_innov(i);
            if par_gene_idx.is_some() {
                let par_gene = parent.edge_genes[par_gene_idx].clone();
                par_edgeome.push(EdgeLocus::Gene(par_gene));
            } else {
                par_edgeome.push(EdgeLocus::Gap);
            }
        }

        par_edgeome
    }

    // returns node genes in order and with gaps
    fn order_nodes(parent: &Genome, innov_num: usize) -> Vec<NodeGene> {
        let par_nodeome: Vec<EdgeLocus> = Vec::new();

        for i in 0..=max_innov {
            par_gene_idx = parent.node_index_from_innov(i);
            if par_gene_idx.is_some() {
                let par_gene = parent.edge_genes[par_gene_idx].clone();
                par_nodeome.push(EdgeLocus::Gene(par_gene));
            } else {
                par_nodeome.push(EdgeLocus::Gap);
            }

            par_nodeome
        }

        mod tests {
            #[test]
            fn test_from_parents() {
                let mut gen_1 = Genome::new_dense(3, 4);
                let mut gen_2 = Genome::new_dense(3, 5);

                // remove a couple of genes
                gen_1._remove_by_innovation(3);
                gen_1._remove_by_innovation(4);
                gen_1._remove_by_innovation(6);
                gen_2._remove_by_innovation(6);
                gen_2._remove_by_innovation(11);

                // we need to construct some known edgeomes and nodeomes
                // easy to write down indices and we'll fill the -omes in loops
                let known_1_edge_idx = vec![0, 1, 2, -1, -1, 5, -1, 7, 8, 9, 10, 11, -1, -1, -1];
                let known_2_edge_idx = vec![0, 1, 2, 3, 4, 5, -1, 7, 8, 9, 10, -1, 12, 13, 14];

                // declare our knowns
                let edges_1: Vec<EdgeGene>;
                let edges_2: Vec<EdgeGene>;

                for i in 0..known_1_edge_idx.len() {
                    if known_1_edge_idx[i] >= 0 {
                        let gene = EdgeGene::Gene(Edge::new_dummy(known_1_edge_idx[i]));
                        edges_1.push(gene);
                    } else {
                        edges_1.push(EdgeGene::Gap)
                    }

                    if known_2_edge_idx[i] >= 0 {
                        let gene = EdgeGene::Gene(Edge::new_dummy(known_2_edge_idx[i]));
                        edges_2.push(gene);
                    } else {
                        edge_2.push(EdgeGene::Gap)
                    }
                }

                let known_1_node_idx = vec![0, 1, 2, 3, 4, 5, 6, -1];
                let known_2_node_idx = vec![0, 1, 2, 3, 4, 5, 6, 7];

                let nodes_1: Vec<NodeGene>;
                let nodes_2: Vec<NodeGene>;
                for i in 0..known_1_node_idx.len() {
                    if known_1_node_idx[i] >= 0 {
                        let gene = NodeGene::Gene(Node::new(known_1_node_idx[i], 0., 0.));
                        nodes_1.push(gene);
                    } else {
                        nodes_1.push(NodeGene::Gap)
                    }

                    if known_2_node_idx[i] >= 0 {
                        let gene = NodeGene::Gene(Node::new(known_2_node_idx[i], 0., 0.));
                        nodes_2.push(gene);
                    } else {
                        nodes_2.push(NodeGene::Gap)
                    }
                }

                let aln = Alignment::from_parents(gen_1, gen_2);

                assert_eq!(known_1_edges, aln.edgeomes[0]);
                assert_eq!(known_2_edges, aln.edgeomes[1]);
                assert_eq!(known_1_nodes, aln.nodeomes[0]);
                assert_eq!(known_2_nodes, aln.nodeomes[1]);
                assert_eq!(14, aln.max_edge_innov);
                assert_eq!(7, aln.max_node_innov)
            }
        }
    }
}
