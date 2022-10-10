#[derive(Clone)]
#[derive(PartialEq)]
pub struct Edge {
    pub innovation: usize,
    pub source_i: usize,
    pub target_i: usize,
    pub weight: f64,
    pub enabled: bool,
}

impl Edge {
    pub fn new(innovation: usize, source_i: usize, target_i: usize, weight: f64) -> Edge {
        Edge {
            innovation,
            source_i,
            target_i,
            weight,
            enabled: true,
        }
    }
}