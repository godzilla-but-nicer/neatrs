pub struct CommunityParams {
    pub disjoint_imp: f64,
    pub excess_imp: f64,
    pub weight_imp: f64,
    pub species_thresh: f64,
}

impl CommunityParams {
    pub fn new() -> CommunityParams {
        CommunityParams {
            disjoint_imp: 1.,
            excess_imp: 1.,
            weight_imp: 1.,
            species_thresh: 1.5,
        }
    }
}