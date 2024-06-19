use crate::community::Genome;
use crate::neural_network::edge::Edge;
use crate::neural_network::node::Node;

use std::error::Error;
use rand::{prelude, Rng};

use super::{innovation_tracker::InnovationTracker, genome::GenomeParams};


/// Genomes in alignments are actually made up of Loci which keep track of both genes and lack of
/// genes. Genes contain either Nodes or Edges.
#[derive(Clone, Debug, PartialEq)]
pub enum Locus<T: Clone> {
    Gene(T),
    Gap,
}

impl<T: Clone> Locus<T> {
    /// returns true if the Locus is a Gap
    fn is_gap(&self) -> bool {
        match self {
            Locus::Gap => true,
            Locus::Gene(_) => false
        }
    }

    /// Returns the gene in a Gene locus or panics if called on a Gap
    fn get_gene(&self) -> T {
        match self {
            Locus::Gene(x) => x.clone(),
            Locus::Gap => panic!("Attempt to `get_gene()` on Locus::Gap"),
        }
    }
}

/// Return enum for many of the gene matching functions. Mismatch indicates one of the genomes is
/// missing the focal gene
#[derive(Clone, Debug, PartialEq)]
pub enum PotentialMismatch {
    Match(Option<f64>), // float represents weight difference
    Mismatch,
}

/// Allows simple pattern matching for pretty syntax when handling crossover
#[derive(Debug, Clone)]
pub enum CrossoverMode {
    Alternating,
    SimpleRandom,
}

/// This struct simply holds any relevant parameters for alignment related stuff
#[derive(Debug)]
pub struct AlignmentParams {
    pub crossover_mode: CrossoverMode,
}

/// Provides named fields for the incompatability numbers so we don't get confused elsewhere.
/// Weighing and summing these values occurs in Species
#[derive(Clone, Debug, PartialEq)]
pub struct IncompatabilityComponents {
    pub base_disjoint: f64,
    pub base_excess: f64,
    pub base_weight_diff: f64,
}

/// Tracks important nodes whose identities do not change
#[derive(Clone, Debug)]
struct GenomeStructure {
    sensor_innovs: Vec<usize>,
    output_innovs: Vec<usize>,
}

/// This module handles all of the logic associated with comparing genomes gene-for gene. This need
/// arrises in at least two different places. First, for comparing the similarity of genomes for
/// assigning species groupings. Second, for creating new genomes through the recombination of two
/// parent genomes.
/// 
/// This object holds sorted vectors of Loci in innovation number order with Gaps inserted such that an
/// index refers to the same innovation number or gap for each genome. These vectors for both Nodes
/// and Edges are the core data held by Alignments.
#[derive(Clone, Debug)]
pub struct Alignment<'a> {
    // actual objects upon which recombination acts
    edgeomes: Vec<Vec<Locus<Edge>>>,
    nodeomes: Vec<Vec<Locus<Node>>>,

    // max innovation numbers
    max_edge_innov: usize,
    max_node_innov: usize,

    // used in calculating incompatability
    excess_locus: usize,

    // indices of importance
    structure: GenomeStructure,

    // behavioral parameters
    params: &'a AlignmentParams,
}

impl <'a> Alignment<'a> {
    /// The primary constructor for Alignments takes two genomes representing the parents
    pub fn from_parents(p0: &Genome, p1: &Genome, params: &'a AlignmentParams) -> Alignment<'a> {
        // alignment vectors of the two genomes
        let mut p0_innovs = Vec::new();
        let mut p1_innovs = Vec::new();

        // pull out the innovation numbers from edges
        for p0_edge in &p0.edge_genes {
            p0_innovs.push(p0_edge.innov);
        }

        for p1_edge in &p1.edge_genes {
            p1_innovs.push(p1_edge.innov);
        }

        // we need to fill any disjoint or excess edges with Gaps
        let max_vals = vec![
            *p0_innovs.iter().max().unwrap(),
            *p1_innovs.iter().max().unwrap(),
        ];
        let max_edge_innov = *max_vals.iter().max().unwrap();
        let excess_locus = *max_vals.iter().min().unwrap();

        // these hold the loci of the genomes that will be aligned
        let p0_edgeome = Self::order_edges(&p0, max_edge_innov);
        let p1_edgeome = Self::order_edges(&p1, max_edge_innov);

        let edgeomes = vec![p0_edgeome, p1_edgeome];

        // Now we have to handle the nodes with the same proceedure
        let mut p0_node_innovs: Vec<usize> = Vec::new();
        let mut p1_node_innovs: Vec<usize> = Vec::new();

        // get the max innovation number again
        for p0_node in &p0.node_genes {
            p0_node_innovs.push(p0_node.innov)
        }

        for p1_node in &p1.node_genes {
            p1_node_innovs.push(p1_node.innov)
        }

        let max_node_vals = vec![
            *p0_node_innovs.iter().max().unwrap(),
            *p1_node_innovs.iter().max().unwrap(),
        ];
        let max_node_innov = *max_node_vals.iter().max().unwrap();

        // get the aligned nodeomes
        let p0_nodeome = Self::order_nodes(&p0, max_node_innov);
        let p1_nodeome = Self::order_nodes(&p1, max_node_innov);

        let nodeomes = vec![p0_nodeome, p1_nodeome];

        // pull out indices we need to care about
        let structure = GenomeStructure {
            sensor_innovs: p0.sensor_innovs.clone(),
            output_innovs: p0.output_innovs.clone() 
        };

        Alignment {
            edgeomes,
            nodeomes,
            max_edge_innov,
            excess_locus,
            max_node_innov,
            structure,
            params,
        }
    }

    /// This calculates the incompatibility components in the edges number of excess genes---genes
    /// in one genome with higher innovation numbers than contained in the partner genome, disjoint
    /// genes---where only one genome contains a particular innovation numbered gene but not due to
    /// simply "more evolution" in that genome, and weight differences---the average difference in
    /// weights for genes contained in both genomes. 
    pub fn raw_incompatibility(p0: &Genome, p1: &Genome, params: &'a AlignmentParams) -> IncompatabilityComponents {
        // align to count disjoint and excess genes
        let alignment = Self::from_parents(&p0, &p1, params);

        // iterate over edges and get raw incompatibility values
        let mut internal_mismatches = 0;
        let mut external_mismatches = 0;
        let mut diffs = 0.0;
        let mut diff_norm = 0.0;


        for locus in 0..alignment.max_edge_innov {

            // if the locus is in the area we count as "disjoint"
            if locus < alignment.excess_locus {
            
                match alignment.check_mismatch(locus) {
            
                    PotentialMismatch::Mismatch => internal_mismatches += 1,
                    PotentialMismatch::Match(optional_diff) => match optional_diff {
            
                        Some(diff) => {
                            diffs += diff;
                            diff_norm += 1.0;
                        }
                        None => continue
            
                    }
                };

            // if the locus is in the area we could as "excess"
            } else {
                external_mismatches += 1;
            }
        }

        // normalize
        let external_norm = external_mismatches as f64 / alignment.edgeomes[0].len() as f64;
        let internal_norm = internal_mismatches as f64 / alignment.edgeomes[0].len() as f64;
        let avg_diff = diffs / diff_norm;

        IncompatabilityComponents {
            base_disjoint: internal_norm,
            base_excess: external_norm,
            base_weight_diff: avg_diff,
        }
    }

    /// Helper function for counting incompatibility components by checking whether a locus is a
    /// match or not.
    fn check_mismatch(&self, locus: usize) -> PotentialMismatch {
        let mut p0_weight = 0.;
        let mut p1_weight = 0.;

        let p0_missing = match &self.edgeomes[0][locus] {
            Locus::Gap => true,
            Locus::Gene(g) => {
                p0_weight = g.weight;
                false
            }
        };

        let p1_missing = match &self.edgeomes[1][locus] {
            Locus::Gap => true,
            Locus::Gene(g) => {
                p1_weight = g.weight;
                false
            }
        };

        if p0_missing ^ p1_missing {
            PotentialMismatch::Mismatch
        } else if !p0_missing & !p1_missing {
            PotentialMismatch::Match(Some((p0_weight - p1_weight).abs()))
        } else {
            PotentialMismatch::Match(None)
        }
    }

    /// Helper function for identifying a Locus for which neither edgeome has a Gene
    fn edgeome_double_gap(&self, locus: usize) -> bool {

        if self.edgeomes[0][locus].is_gap() & self.edgeomes[1][locus].is_gap() {
            true
        } else {
            false
        }
    
    }

    /// Boolean for whether a gene is found only in the first parent's edgeome
    fn edgeome_p0_only(&self, locus: usize) -> bool {

        if !self.edgeomes[0][locus].is_gap() & self.edgeomes[1][locus].is_gap() {
            true
        } else {
            false
        }

    }

    /// Boolean for whether a gene is found only in the second parent's edgeome
    fn edgeome_p1_only(&self, locus: usize) -> bool {
        
        if self.edgeomes[0][locus].is_gap() & !self.edgeomes[1][locus].is_gap() {
            true
        } else {
            false
        }
    
    }
    
    /// Boolean for whether a gene is found only in the first parent's nodeome
    fn nodeome_p0_only(&self, locus: usize) -> bool {

        if !self.nodeomes[0][locus].is_gap() & self.nodeomes[1][locus].is_gap() {
            true
        } else {
            false
        }

    }

    /// Boolean for whether a gene is found only in the second parent's nodeome
    fn nodeome_p1_only(&self, locus: usize) -> bool {
        
        if self.nodeomes[0][locus].is_gap() & !self.nodeomes[1][locus].is_gap() {
            true
        } else {
            false
        }
    }

    /// Order's the Loci of a parent's edgeome by innovation number and inserts Gaps where needed
    fn order_edges(parent: &Genome, innov_num: usize) -> Vec<Locus<Edge>> {
        let mut par_edgeome: Vec<Locus<Edge>> = Vec::new();

        for i in 0..=innov_num {
            match parent.edge_index_from_innov(i) {
                None => par_edgeome.push(Locus::Gap),
                Some(x) => {
                    let par_gene = parent.edge_genes[x].clone();
                    par_edgeome.push(Locus::Gene(par_gene))
                }
            };
        }

        par_edgeome
    }

    /// Order's the Loci of a parent's edgeome by innovation number and inserts Gaps where needed
    fn order_nodes(parent: &Genome, innov_num: usize) -> Vec<Locus<Node>> {
        let mut par_nodeome: Vec<Locus<Node>> = Vec::new();

        for i in 0..=innov_num {
            match parent.node_index_from_innov(i) {
                None => par_nodeome.push(Locus::Gap),
                Some(x) => {
                    let par_gene = parent.node_genes[x].clone();
                    par_nodeome.push(Locus::Gene(par_gene))
                }
            };
        }

        par_nodeome
    }

    /// Reads the alignment parameters to call the correct crossover method
    pub fn crossover(p0: Genome, p1: Genome, params: &AlignmentParams) -> Genome {

        let alignment = Alignment::from_parents(&p0, &p1, params);

        let new_genome = match alignment.params.crossover_mode {
            CrossoverMode::Alternating => alignment.alternating_crossover(),
            CrossoverMode::SimpleRandom => alignment.simple_random_crossover(),
        };

        new_genome
    }

    /// This crossover function simply takes the first gene from the first parent, the second from
    /// the second, the third from the first, and so on for both nodes and edges to construct a new
    /// genome. If only one parent has a Gene for a given Locus the child gets that gene.
    fn alternating_crossover(&self) -> Genome {
        
        let mut new_nodes: Vec<Node> = Vec::new();
        let mut new_edges = Vec::new();

        let mut p0_is_donor = true;

        // iterate over edges
        for locus in 0..self.edgeomes[0].len() {

            if !self.edgeomes[0][locus].is_gap() & p0_is_donor {

                new_edges.push(self.edgeomes[0][locus].get_gene());
                p0_is_donor = false;
                
            } else if !self.edgeomes[1][locus].is_gap() & !p0_is_donor {

                new_edges.push(self.edgeomes[1][locus].get_gene());
                p0_is_donor = true;

            } else if self.edgeome_p0_only(locus) {

                new_edges.push(self.edgeomes[0][locus].get_gene());
                p0_is_donor = false;

            } else if self.edgeome_p1_only(locus) {

                new_edges.push(self.edgeomes[1][locus].get_gene());
                p0_is_donor = true;
            
            }
        }

        // iterate over nodeomes
        for locus in 0..self.nodeomes[0].len() {

            if !self.edgeomes[0][locus].is_gap() & p0_is_donor {

                new_nodes.push(self.nodeomes[0][locus].get_gene());
                p0_is_donor = false;
                
            } else if !self.nodeomes[1][locus].is_gap() & !p0_is_donor {

                new_nodes.push(self.nodeomes[1][locus].get_gene());
                p0_is_donor = true;

            } else if self.nodeome_p0_only(locus) {

                new_nodes.push(self.nodeomes[0][locus].get_gene());
                p0_is_donor = false;

            } else if self.nodeome_p1_only(locus) {

                new_nodes.push(self.nodeomes[1][locus].get_gene());
                p0_is_donor = true;
            
            }
        }
        
        Genome::new(new_nodes, new_edges, 
                    self.structure.sensor_innovs.clone(), 
                    self.structure.output_innovs.clone())

    }

    /// This crossover function constructs a child Genome by randomly selecting a parent for each
    /// gene unless only one parent has a gene for the Locus of interest. If only one parent has a
    /// Gene at a given Locus, the child inherits that gene. 
    fn simple_random_crossover(&self) -> Genome {

        let mut rng = rand::thread_rng();

        let mut new_nodes = Vec::new();
        let mut new_edges = Vec::new();

        // select random parents for nodes
        for locus in 0..self.nodeomes[0].len() {
            
            // which parent donates if possible
            let roll: usize = rng.gen_range(0..=1);

            if !self.nodeomes[roll][locus].is_gap() {
                new_nodes.push(self.nodeomes[roll][locus].get_gene())
            } else if !self.nodeomes[1 - roll][locus].is_gap() {
                new_nodes.push(self.nodeomes[1 - roll][locus].get_gene())
            }

        }

        // repeat for edge genes
        for locus in 0..self.edgeomes[0].len() {
            
            // which parent donates if possible
            let roll: usize = rng.gen_range(0..=1);

            if !self.edgeomes[roll][locus].is_gap() {
                new_edges.push(self.edgeomes[roll][locus].get_gene())
            } else if !self.edgeomes[1 - roll][locus].is_gap() {
                new_edges.push(self.edgeomes[1 - roll][locus].get_gene())
            }

        }

        Genome::new(new_nodes, new_edges,
                self.structure.sensor_innovs.clone(),
                self.structure.output_innovs.clone())
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    fn test_params() -> AlignmentParams {
        AlignmentParams { crossover_mode:CrossoverMode::Alternating }
    }


    #[rstest]
    fn test_from_parents(test_params: AlignmentParams) {
        let mut gen_1 = Genome::new_dense(2, 2);
        let mut gen_2 = Genome::new_dense(2, 2);

        // remove a couple of genes
        gen_1._remove_by_innovation(1);
        gen_1._remove_by_innovation(2);
        gen_2._remove_by_innovation(1);
        gen_2._remove_by_innovation(3);

        // we need to construct some known edgeomes and nodeomes
        let edges_1 = vec![Locus::Gene(Edge::new(0, 0, 2, 0.0)),
                                             Locus::Gap,
                                             Locus::Gap,
                                             Locus::Gene(Edge::new(3, 1, 3, 0.0))];
        let edges_2 = vec![Locus::Gene(Edge::new(0, 0, 2, 0.0)),
                                             Locus::Gap,
                                             Locus::Gene(Edge::new(2, 1, 2, 0.0)),
                                             Locus::Gap];

        let nodes_1 = vec![Locus::Gene(Node::new(0, 0.0, Node::linear)),
                                             Locus::Gene(Node::new(1, 0.0, Node::linear)),
                                             Locus::Gene(Node::new(2, 0.0, Node::linear)),
                                             Locus::Gene(Node::new(3, 0.0, Node::linear))];
        let nodes_2 = vec![Locus::Gene(Node::new(0, 0.0, Node::linear)),
                                             Locus::Gene(Node::new(1, 0.0, Node::linear)),
                                             Locus::Gene(Node::new(2, 0.0, Node::linear)),
                                             Locus::Gene(Node::new(3, 0.0, Node::linear))];

        let aln = Alignment::from_parents(&gen_1, &gen_2, &test_params);

        println!("Edges");
        for gene_i in 0..aln.edgeomes[0].len() {
            match aln.edgeomes[0][gene_i] {
                Locus::Gene(_) => println!("(p1, locus {}): Gene", gene_i),
                Locus::Gap => println!("(p1, locus {}): Gap", gene_i)
            }
            match aln.edgeomes[1][gene_i] {
                Locus::Gene(_) => println!("(p2, locus {}):      Gene", gene_i),
                Locus::Gap => println!("(p2, locus {}):      Gap", gene_i)
            }
        }
        println!("Nodes");
        for gene_i in 0..aln.nodeomes[0].len() {
            match aln.nodeomes[0][gene_i] {
                Locus::Gene(_) => println!("(p1, locus {}): Gene", gene_i),
                Locus::Gap => println!("p1, locus {}: Gap", gene_i)
            }
            match aln.nodeomes[1][gene_i] {
                Locus::Gene(_) => println!("(p2, locus {}):      Gene", gene_i),
                Locus::Gap => println!("(p2, locus {}):      Gap", gene_i)
            }
        }

        assert_eq!(edges_1, aln.edgeomes[0]);
        assert_eq!(edges_2, aln.edgeomes[1]);
        assert_eq!(nodes_1, aln.nodeomes[0]);
        assert_eq!(nodes_2, aln.nodeomes[1]);
        assert_eq!(3, aln.max_edge_innov);
        assert_eq!(3, aln.max_node_innov)
    }

    #[rstest]
    fn test_raw_incompatibility() {

        let known = IncompatabilityComponents {
            base_disjoint: 2.0 / 6.0,
            base_excess: 1.0 / 6.0,
            base_weight_diff: 2.0 / 2.0,
        };

        let mut gen_1 = Genome::new_dense(2, 3);
        let mut gen_2 = Genome::new_dense(2, 3);

        // set the weights of some of the genes
        gen_1.edge_genes[0].weight = 1.0;
        gen_2.edge_genes[4].weight = 1.0;
        
        // remove a couple of genes
        gen_1._remove_by_innovation(1);
        gen_1._remove_by_innovation(2);
        gen_1._remove_by_innovation(3);
        gen_2._remove_by_innovation(1);
        gen_2._remove_by_innovation(5);


        assert_eq!{known, Alignment::raw_incompatibility(&gen_1, &gen_2, &test_params())}
    }
}
