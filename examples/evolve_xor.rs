use neatrs::NEAT;
use neatrs::neural_network::NeuralNetwork;
use neatrs::community::genome::Genome;

use ndarray::{Array, Array1};
use rand::prelude::*;
use rand::seq::SliceRandom;

use std::io::{prelude::*, BufWriter};
use std::fs::File;

fn main() {
    struct XorReport {
        time: Array1<f64>,
        input_1: Array1<f64>,
        input_2: Array1<f64>,
        target: Array1<f64>,
        output: Array1<f64>,
        fitness: f64
    }

    fn xor(genome: &Genome) -> XorReport {
        
        // many programs will have a time aspect
        let start_time = 0.0;
        let end_time = 5.0;
        let time_step = 0.1;
        
        let time = Array1::<f64>::range(start_time, end_time, time_step);
        
        let mut output = Array1::<f64>::zeros(time.dim());

        let mut input_1 = Array1::<f64>::zeros(time.dim());
        let mut input_2 = Array1::<f64>::zeros(time.dim());

        let mut target = Array1::<f64>::zeros(time.dim());


        // randomly generate inputs and target outputs changing every second
        let mut rng = thread_rng();

        let mut sensor_1: bool = false;
        let mut sensor_2: bool = false;
        for i in 0..time.dim() {
            if time[i].fract() == 0.0 {
                sensor_1 = rng.gen();
                sensor_2 = rng.gen();
            }
            input_1[i] = sensor_1 as u8 as f64;
            input_2[i] = sensor_2 as u8 as f64;

            target[i] = (sensor_1 ^ sensor_2) as u8 as f64;
        }

        // construct neural networks and run for time duration
        let mut nn = NeuralNetwork::from_genome(genome);

        for t in 0..time.dim() {
            nn.propagate(vec![input_1[t], input_2[t]]);
            output[t] = nn.get_output()[0];
        }

        let absolute_diff = (&output - &target).mapv(f64::abs);
        let fitness = absolute_diff.mapv(|x| (-1. * x).exp()).sum();

        XorReport {
            time,
            input_1,
            input_2,
            target,
            output,
            fitness
        }
    }

    fn xor_fitness(genome: &Genome) -> f64 {
        let report = xor(genome);

        report.fitness
    }

    fn write_run_report(path: &str, run: XorReport) {

        let file = File::create(path).unwrap();
        let mut writer = BufWriter::new(file);

        writeln!(&mut writer, "time,input_1,input_2,output,target").unwrap();

        for i in 0..run.time.dim() {
            writeln!(&mut writer, "{},{},{},{},{}", run.time[i], run.input_1[i], run.input_2[i], run.output[i], run.target[i]).unwrap();
        }
    }

    let mut neat = NEAT::from_parameters("examples/parameters/evolve_xor.yaml",
                                           xor_fitness);
    
    println!("Number of genomes at init: {}", neat.number_of_genomes());
    neat.evolve(10);
    println!("Number of genomes at gen 10: {}", neat.number_of_genomes());
    neat.write_fitness_values("examples/data/evolve_xor/fitness_10.txt");
    let champion = neat.get_champion();

    let report10 = xor(&champion);

    write_run_report("examples/data/evolve_xor/run_10.csv", report10);
    
    neat.evolve(10);
    neat.write_fitness_values("examples/data/evolve_xor/fitness_20.txt");
    let champion = neat.get_champion();

    let report20 = xor(&champion);

    write_run_report("examples/data/evolve_xor/run_20.csv", report20);
    
    neat.evolve(10);
    neat.write_fitness_values("examples/data/evolve_xor/fitness_30.txt");
    let champion = neat.get_champion();

    let report30 = xor(&champion);

    write_run_report("examples/data/evolve_xor/run_30.csv", report30);

}