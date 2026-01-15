use rand::Rng;

use crate::matrix::matrix::Matrix;

fn generate_parameter(rows: usize, cols: usize, min: f32, max: f32) -> Matrix<f32> {
    let mut rng = rand::thread_rng();
    let mut data: Vec<f32> = vec![];

    for _ in 0..(rows * cols) {
        data.push(rng.gen_range(min..max));
    }

    Matrix {
        rows: rows,
        cols: cols,
        data: data
    }
}


pub trait Layer {
    fn calculate_gradients(&mut self, previous_gradients: Vec<Matrix<f32>>) -> Vec<Matrix<f32>>;
    fn adjust_parameters(&mut self, learning_rate: f32);
}


pub struct Dense {
    pub weights: Matrix<f32>,
    pub weights_gradients: Option<Matrix<f32>>,

    pub biases: Matrix<f32>,
    pub biases_gradients: Option<Matrix<f32>>,

    pub previous_input: Option<Matrix<f32>>
}

impl Dense {
    pub fn new(nodes: usize, input_size: usize, parameter_min: f32, parameter_max: f32) -> Dense {
        Dense {
            weights: generate_parameter(input_size, nodes, parameter_min, parameter_max),
            weights_gradients: None,

            biases: generate_parameter(1, nodes, parameter_min, parameter_max),
            biases_gradients: None,

            previous_input: None
        }    
    }

    fn compute(&mut self, input: Matrix<f32>, handle_gradients: bool) -> Matrix<f32> {
        let c_input = input;
        
        if handle_gradients {
            self.previous_input = Some(c_input.clone());
        }

        c_input * self.weights.clone() + self.biases.clone()
    }
}

impl Layer for Dense {
    fn calculate_gradients(&mut self, previous_gradients: Vec<Matrix<f32>>) -> Vec<Matrix<f32>> {
        if self.previous_input.is_none() {
            println!("Dense Error - Previos input is none.");
            return vec![];
        }

        self.weights_gradients = Some(self.previous_input.clone().unwrap().transpose() * previous_gradients[0].clone());
        self.biases_gradients = Some(previous_gradients[0].clone());


        return vec![previous_gradients[0].clone() * self.weights.clone().transpose()];
    }


    fn adjust_parameters(&mut self, learning_rate: f32) {
        self.weights = self.weights.clone() - self.weights_gradients.clone().unwrap() * learning_rate;
        self.biases = self.biases.clone() - self.biases_gradients.clone().unwrap() * learning_rate;
    }
}