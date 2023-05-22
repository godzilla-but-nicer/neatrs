
#[derive(Debug)]
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
            species_thresh: 1.3,
        }
    }
}

#[derive(Debug)]
pub struct SpeciesParams {
    pub mate_fraction: f64,
    pub num_elites: usize,
    pub prob_structural: f64,
    pub crossover_mode: CrossoverMode,
}

#[derive(Debug)]
#[derive(PartialEq)]
pub enum CrossoverMode {
    SimpleRandom,
    Alternating,
}

impl SpeciesParams {
    pub fn new() -> SpeciesParams {
        SpeciesParams {
            mate_fraction: 10.0,
            num_elites: 3,
            prob_structural: 0.3,
            crossover_mode: CrossoverMode::SimpleRandom,
        }
    }

    pub fn get_test_params() -> SpeciesParams {
        SpeciesParams {
            mate_fraction: 10.0,
            num_elites: 1,
            prob_structural: 0.3,
            crossover_mode: CrossoverMode::Alternating,
        }
    }
}

#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub struct GenomeParams {
    pub disjoint_imp: f64,
    pub excess_imp: f64,
    pub weight_imp: f64,
    pub insert_prob: f64,
    pub connect_prob: f64,
    pub weight_mut_sd: f64,
    pub bias_mut_sd: f64,
}

impl GenomeParams {
    pub fn new() -> GenomeParams {
        GenomeParams {
            disjoint_imp: 1.,
            excess_imp: 1.,
            weight_imp: 0.2,
            insert_prob: 0.5,
            connect_prob: 0.5,
            weight_mut_sd: 1.0,
            bias_mut_sd: 1.0,
        }
    }
    
    pub fn get_test_params() -> GenomeParams {
        GenomeParams {
            disjoint_imp: 1.,            
            excess_imp: 1.,
            weight_imp: 0.1,
            insert_prob: 0.4,
            connect_prob: 0.3,
            weight_mut_sd: 1.0,
            bias_mut_sd: 1.0,
        }
    }
}