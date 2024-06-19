/// The edge struct is one of the basic units of organization for neural networks. Here we
/// store all of the relevant information needed to calculate node and network activations.
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub struct Edge {
    pub innov: usize,
    pub source_innov: usize,
    pub target_innov: usize,
    pub weight: f64,
    pub enabled: bool,
}

impl Edge {
    /// Basic constructor for edges. Requires all information and defaults to enabled state.
    pub fn new(innov: usize, source_innov: usize, target_innov: usize, weight: f64) -> Edge {
        Edge {
            innov,
            source_innov,
            target_innov,
            weight,
            enabled: true,
        }
    }

    pub fn new_dummy(innov: usize) -> Edge {
        Edge {
            innov,
            source_innov: 0,
            target_innov: 0,
            weight: 0.0,
            enabled: true,
        }
    }
}