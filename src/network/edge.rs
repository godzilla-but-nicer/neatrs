
#[derive(PartialEq)]
pub struct Edge {
    pub source_i: usize,
    pub target_i: usize,
    pub weight: f64,
}

impl Edge {
    pub fn new(source_i: usize, target_i: usize, weight: f64) -> Edge {
        Edge {
            source_i,
            target_i,
            weight
        }
    }
}