use crate::community::mutation::MutationInfo;

#[derive(Clone)]
pub struct InnovationTracker {
    pub node_max_innov: usize,
    pub edge_max_innov: usize,
}

impl InnovationTracker {

    fn new() -> InnovationTracker {
        InnovationTracker {
            node_max_innov: 0,
            edge_max_innov: 0,
        }
    }

    pub fn update(&self, mutations: &MutationInfo) -> InnovationTracker {

        match mutations {
            MutationInfo::Quantitative(_) => self.clone(),
            MutationInfo::Topological((_, changes)) => {

                let node_max_innov = self.node_max_innov + changes.nodes_added;
                let edge_max_innov = self.edge_max_innov + changes.edges_added;

                InnovationTracker {
                    node_max_innov,
                    edge_max_innov,
                }
            }
        }
    }
}