#![warn(missing_docs)]

//! its NEAT, baby

/// This module contains all of the neural network stuff
mod neural_network;

// This modual contains all of the evolutionary stuff
mod community;

/// Adds two numbers together
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
