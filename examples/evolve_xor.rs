use neatrs::NEAT;
use neatrs::neural_network::NeuralNetwork;
use neatrs::community::genome::Genome;

use ndarray::{Array, Array1};
use rand::prelude::*;
use rand::seq::SliceRandom;

fn main() {
    fn xor_fitness(genome: &Genome) -> f64 {
        
        // many programs will have a time aspect
        let start_time = 0.0;
        let end_time = 5.0;
        let time_step = 0.1;
        
        let times = Array1::<f64>::range(start_time, end_time, time_step);
        
        let mut outputs = Array1::<f64>::zeros(times.dim());

        let mut input_1 = Array1::<f64>::zeros(times.dim());
        let mut input_2 = Array1::<f64>::zeros(times.dim());

        let mut target = Array1::<f64>::zeros(times.dim());


        // randomly generate inputs and target outputs changing every second
        let mut rng = thread_rng();

        let mut sensor_1: bool = false;
        let mut sensor_2: bool = false;
        for i in 0..times.dim() {
            if times[i].fract() == 0.0 {
                sensor_1 = rng.gen();
                sensor_2 = rng.gen();
            }
            input_1[i] = sensor_1 as u8 as f64;
            input_2[i] = sensor_2 as u8 as f64;

            target[i] = (sensor_1 ^ sensor_2) as u8 as f64;
        }

        // construct neural networks and run for time duration
        let mut nn = NeuralNetwork::from_genome(genome);

        for t in 0..times.dim() {
            nn.propagate(vec![input_1[t], input_2[t]]);
            outputs[t] = nn.get_output()[0];
        }

        let absolute_diff = (&outputs - &target).mapv(f64::abs);
        absolute_diff.mapv(|x| (-1. * x).exp()).sum()
    }
}