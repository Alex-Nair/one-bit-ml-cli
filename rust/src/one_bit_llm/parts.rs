use nalgebra::*;
use std::f64::consts::PI;


pub trait Layer {
    fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>);
    fn adjust_parameters(&mut self, learning_rate: f64);
}


pub struct Dense {
    pub weights: DMatrix<f64>,
    pub weight_gradients: DMatrix<f64>,

    pub quantized_weights: DMatrix<f64>,
    
    pub biases: DMatrix<f64>,
    pub bias_gradients: DMatrix<f64>,

    prev_input: DMatrix<f64>
}

impl Dense {
    pub fn calculate(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        self.prev_input = input;
        &self.weights * &self.prev_input + &self.biases
    }
}

impl Layer for Dense {
    fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>) {
        self.weight_gradients = &previous_gradient * &self.prev_input.transpose();
        self.bias_gradients = previous_gradient.clone();
    }
    
    fn adjust_parameters(&mut self, learning_rate: f64) {
        self.weights -= &self.weight_gradients * learning_rate;
        self.biases -= &self.bias_gradients * learning_rate;
    }
}


pub struct GELU {}

impl GELU {
    pub fn calculate(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        let mut cubed_input: DMatrix<f64> = input.clone();

        for row in 0..cubed_input.shape().0 {
            for column in 0..cubed_input.shape().1 {
                cubed_input[(row, column)] = cubed_input[(row, column)].powi(3);
            }
        }

        let mut inside_result: DMatrix<f64> = (&input + 0.044715 * cubed_input) * (2.0 / (PI));
        let one_matrix: DMatrix<f64> = DMatrix::from_element(inside_result.shape().0, inside_result.shape().1, 1.0);

        for row in 0..inside_result.shape().0 {
            for column in 0..inside_result.shape().1 {
                inside_result[(row, column)] = inside_result[(row, column)].tanh();
            }
        }


        0.5 * &input * (one_matrix + inside_result)
    }
}