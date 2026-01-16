use std::{f32::consts::PI};

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


pub struct GELU {
    pub previous_input: Option<Matrix<f32>>
}

impl GELU {
    pub fn new() -> GELU {
        GELU {
            previous_input: None
        }
    }

    fn compute(&mut self, input: Matrix<f32>, handle_gradients: bool) -> Matrix<f32> {
        if handle_gradients {
            self.previous_input = Some(input.clone());
        }

        input.clone() * (((input.clone() +  input.clone().pow_unit(3.0) * 0.044715) * (2.0 / PI).sqrt()).tanh() + 1.0) * 0.5
    }
}

impl Layer for GELU {
    fn calculate_gradients(&mut self, previous_gradients: Vec<Matrix<f32>>) -> Vec<Matrix<f32>> {
        if self.previous_input.is_none() {
            println!("GELU Error - Previos input is none.");
            return vec![];
        }

        const ALPHA_CONSTANT: f32 = 0.044714;
        let one_matrix: Matrix<f32> = Matrix::new(self.previous_input.as_ref().unwrap().rows, self.previous_input.as_ref().unwrap().cols, 1.0);
        
        let f: Matrix<f32> = self.previous_input.clone().unwrap() * 0.5;
        let g: Matrix<f32> = (self.previous_input.clone().unwrap() +  self.previous_input.clone().unwrap().pow_unit(3.0) * ALPHA_CONSTANT * (2.0 / PI).sqrt()).tanh() + 1.0;

        let f_diff: Matrix<f32> = one_matrix.clone() * 0.5;
        let g_diff: Matrix<f32> = (one_matrix.clone()) / ((self.previous_input.clone().unwrap() * (2.0 / PI) + self.previous_input.clone().unwrap().pow_unit(3.0) * (2.0 * ALPHA_CONSTANT / PI)).cosh().pow_unit(2.0)) * (self.previous_input.clone().unwrap().pow_unit(3.0) * (6.0 * ALPHA_CONSTANT / PI) + (2.0 / PI));

        vec![(f_diff * g + f * g_diff) * previous_gradients[0].clone()]
    }

    fn adjust_parameters(&mut self, _learning_rate: f32) {}
}


pub struct FFN {
    pub inner_dense: Dense,
    pub outer_dense: Dense,
    pub activation_layer: GELU
}

impl FFN {
    pub fn new(input_size: usize, inner_size: usize, parameter_min: f32, parameter_max: f32) -> FFN {
        FFN {
            inner_dense: Dense::new(inner_size, input_size, parameter_min, parameter_max),
            outer_dense: Dense::new(input_size, inner_size, parameter_min, parameter_max),
            activation_layer: GELU::new()
        }
    }


    pub fn compute(&mut self, input: Matrix<f32>, handle_gradients: bool) -> Matrix<f32> {
        let result_1: Matrix<f32> = self.inner_dense.compute(input, handle_gradients);
        let result_2: Matrix<f32> = self.activation_layer.compute(result_1, handle_gradients);
        self.outer_dense.compute(result_2, handle_gradients)
    }
}

impl Layer for FFN {
    fn calculate_gradients(&mut self, previous_gradients: Vec<Matrix<f32>>) -> Vec<Matrix<f32>> {
        let result_1: Vec<Matrix<f32>> = self.outer_dense.calculate_gradients(previous_gradients);
        let result_2: Vec<Matrix<f32>> = self.activation_layer.calculate_gradients(result_1);
        self.inner_dense.calculate_gradients(result_2)
    }


    fn adjust_parameters(&mut self, learning_rate: f32) {
        self.inner_dense.adjust_parameters(learning_rate);
        self.outer_dense.adjust_parameters(learning_rate);
    }
}