#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub struct Edge {
    pub innov: usize,
    pub source_i: usize,
    pub target_i: usize,
    pub weight: f64,
    pub enabled: bool,
}

impl Edge {
    pub fn new(innov: usize, source_i: usize, target_i: usize, weight: f64) -> Edge {
        Edge {
            innov,
            source_i,
            target_i,
            weight,
            enabled: true,
        }
    }
}