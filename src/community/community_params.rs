pub struct CommunityParams {
    pub species_thresh: f64,
}

impl CommunityParams {
    pub fn new() -> CommunityParams {
        CommunityParams {
            species_thresh: 1.5,
        }
    }
    
    pub fn get_test_params() -> CommunityParams {
        CommunityParams {
            species_thresh: 1.0,
        }
    }
}

pub struct SpeciesParams {
    mate_percentile: f64
}

impl SpeciesParams {
    pub fn new() -> SpeciesParams {
        SpeciesParams {
            mate_percentile: 90.0,
        }
    }
}

#[derive(Clone)]
#[derive(PartialEq)]
pub struct GenomeParams {
    pub disjoint_imp: f64,
    pub excess_imp: f64,
    pub weight_imp: f64,
}

impl GenomeParams {
    pub fn new() -> GenomeParams {
        GenomeParams {
            disjoint_imp: 1.,
            excess_imp: 1.,
            weight_imp: 0.2,
        }
    }
    
    pub fn get_test_params() -> GenomeParams {
        GenomeParams {
            disjoint_imp: 1.,            
            excess_imp: 1.,
            weight_imp: 0.2,
        }
    }
}