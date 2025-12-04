use nalgebra::*;
use rand::{self, Rng};
use std::f64::consts::PI;


// Helper Functions.
fn pow_matrix(matrix: DMatrix<f64>, pow: i32) -> DMatrix<f64> {
    let mut resultant_matrix: DMatrix<f64> = DMatrix::from_element(matrix.shape().0, matrix.shape().1, 0.0);

    for row in 0..matrix.shape().0 {
        for column in 0..matrix.shape().1 {
            resultant_matrix[(row, column)] = matrix[(row, column)].powi(pow);
        }
    }

    resultant_matrix
}

fn tanh_matrix(matrix: DMatrix<f64>) -> DMatrix<f64> {
    let mut resultant_matrix: DMatrix<f64> = DMatrix::from_element(matrix.shape().0, matrix.shape().1, 0.0);

    for row in 0..matrix.shape().0 {
        for column in 0..matrix.shape().1 {
            resultant_matrix[(row, column)] = matrix[(row, column)].tanh();
        }
    }

    resultant_matrix
}

fn sech_matrix(matrix: DMatrix<f64>) -> DMatrix<f64> {
    let mut resultant_matrix: DMatrix<f64> = DMatrix::from_element(matrix.shape().0, matrix.shape().1, 0.0);

    for row in 0..matrix.shape().0 {
        for column in 0..matrix.shape().1 {
            resultant_matrix[(row, column)] = 1.0 / matrix[(row, column)].cosh();
        }
    }

    resultant_matrix
}

fn element_multiplication_matrix(matrix_1: DMatrix<f64>, matrix_2: DMatrix<f64>) -> DMatrix<f64> {
    let mut resultant_matrix: DMatrix<f64> = DMatrix::from_element(matrix_1.shape().0, matrix_1.shape().1, 0.0);

    for row in 0..matrix_1.shape().0 {
        for column in 0..matrix_1.shape().1 {
            resultant_matrix[(row, column)] = matrix_1[(row, column)] * matrix_2[(row, column)];
        }
    }

    resultant_matrix
}
  


pub trait Layer {
    fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>) -> DMatrix<f64>;
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
    pub fn new(nodes: usize, input_size: usize) -> Self {
        Dense {
            weights: DMatrix::new_random(nodes, input_size),
            weight_gradients: DMatrix::from_element(nodes, input_size, 0.0),
            quantized_weights: DMatrix::from_element(nodes, input_size, 0.0),

            biases: DMatrix::new_random(nodes, 1),
            bias_gradients: DMatrix::from_element(nodes, 1, 0.0),

            prev_input: DMatrix::from_element(nodes, 1, 0.0)
        }
    }

    pub fn calculate(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        self.prev_input = input;
        &self.weights * &self.prev_input + &self.biases
    }
}

impl Layer for Dense {
    fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>) -> DMatrix<f64> {
        self.weight_gradients = &previous_gradient * &self.prev_input.transpose();
        self.bias_gradients = previous_gradient.clone();

        return self.weight_gradients.transpose() * previous_gradient;
    }
    
    fn adjust_parameters(&mut self, learning_rate: f64) {
        self.weights -= &self.weight_gradients * learning_rate;
        self.biases -= &self.bias_gradients * learning_rate;
    }
}


pub struct GELU {}

impl GELU {
    pub fn calculate(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        let inside_result: DMatrix<f64> = (&input + 0.044715 * pow_matrix(input.clone(), 3)) * (2.0 / (PI));
        let one_matrix: DMatrix<f64> = DMatrix::from_element(inside_result.shape().0, inside_result.shape().1, 1.0);

        0.5 * &input * (one_matrix + tanh_matrix(inside_result))
    }
}


pub struct FFN {
    pub d_model: usize,
    pub inner_size: usize,

    pub first_layer: Dense,
    pub second_layer: Dense,
    pub activation_layer: GELU,

    prev_inbetween_output: DMatrix<f64>,
    prev_output_result: DMatrix<f64>
}

impl FFN {
    pub fn new(d_model: usize, inner_size: usize) -> Self {
        FFN {
            d_model: d_model,
            inner_size: inner_size,

            first_layer: Dense::new(inner_size, d_model),
            second_layer: Dense::new(d_model, inner_size),
            activation_layer: GELU {},

            prev_inbetween_output: DMatrix::from_element(inner_size, 1, 0.0),
            prev_output_result: DMatrix::from_element(d_model, 1, 0.0),
        }
    }

    pub fn calculate(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        self.prev_inbetween_output = self.first_layer.calculate(input);
        let activated_output: DMatrix<f64> = self.activation_layer.calculate(self.prev_inbetween_output.clone());
        self.prev_output_result = self.second_layer.calculate(activated_output);

        self.prev_output_result.clone()
    }
}

impl Layer for FFN {
    fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>) -> DMatrix<f64> {
        const FIXED_CONSTANT: f64 = 0.044715;

        let activation_gradients: DMatrix<f64> = self.second_layer.calculate_gradients(previous_gradient.clone());
        let one_matrix: DMatrix<f64> = DMatrix::from_element(activation_gradients.shape().0, activation_gradients.shape().1, 1.0);

        let chained_gradients: DMatrix<f64> = 0.5 * (one_matrix.clone() + tanh_matrix((2.0 / PI) * (activation_gradients.clone() +  FIXED_CONSTANT * pow_matrix(activation_gradients.clone(), 3)))) + 0.5 * activation_gradients.clone() * (((2.0 / PI) * one_matrix.clone() + (((FIXED_CONSTANT * 6.0) / PI) * pow_matrix(activation_gradients.clone(), 2))) * pow_matrix(sech_matrix((2.0 / PI) * one_matrix.clone() + ((2.0 * FIXED_CONSTANT) / PI) * pow_matrix(activation_gradients.clone(), 3)), 2));
        let loss_adjusted_gradients: DMatrix<f64> = element_multiplication_matrix(activation_gradients, chained_gradients);

        let final_gradients: DMatrix<f64> = self.first_layer.calculate_gradients(loss_adjusted_gradients);
        return final_gradients;
    }
    
    fn adjust_parameters(&mut self, learning_rate: f64) {
        self.first_layer.adjust_parameters(learning_rate);
        self.second_layer.adjust_parameters(learning_rate);
    }
}


pub struct LayerNorm {
    pub scaling_parameter: f64,
    pub shifting_parameter: f64,

    scaling_gradient: f64,
    shifting_gradient: f64,

    prev_mean: DMatrix<f64>,
    prev_variance: DMatrix<f64>,

    prev_inputs: DMatrix<f64>,
    prev_normalized_inputs: DMatrix<f64>
}

impl LayerNorm {
    pub fn new(range: f64) -> LayerNorm {
        LayerNorm {
            scaling_parameter: rand::thread_rng().gen_range((-1.0 * range)..range),
            scaling_gradient: 0.0,

            shifting_parameter: rand::thread_rng().gen_range((-1.0 * range)..range),
            shifting_gradient: 0.0,

            prev_mean: DMatrix::from_element(1, 1, 0.0),
            prev_variance: DMatrix::from_element(1, 1, 0.0),

            prev_inputs: DMatrix::from_element(1, 1, 0.0),
            prev_normalized_inputs: DMatrix::from_element(1, 1, 0.0)
        }
    }

    pub fn calculate(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        const EPSILON_CONSTANT: f64 = 0.00001;

        self.prev_inputs = input.clone();

        // Calculating mean and variance.
        let mut means: DMatrix<f64> = DMatrix::from_element(1, input.shape().1, 0.0);
        let mut variances: DMatrix<f64> = DMatrix::from_element(1, input.shape().1, 0.0);
    
        for col in 0..input.shape().1 {
            for row in 0..input.shape().0 {
                means[(0, col)] += input[(row, col)];
            }

            means[(0, col)] /= input.shape().0 as f64;
        }

        self.prev_mean = means.clone();

        for col in 0..input.shape().1 {
            for row in 0..input.shape().0 {
                variances[(0, col)] += (input[(row, col)] - means[(0, col)]).powi(2);
            }

            variances[(0, col)] /= input.shape().0 as f64;
        }

        self.prev_variance = variances.clone();

        // Normalizing Inputs
        let mut normalized: DMatrix<f64> = DMatrix::from_element(input.shape().0, input.shape().1, 0.0);

        for col in 0..input.shape().1 {
            for row in 0..input.shape().0 {
                normalized[(row, col)] = (input[(row, col)] - means[(0, col)]) / (variances[(0, col)] + EPSILON_CONSTANT).sqrt();
            }
        }

        self.prev_normalized_inputs = normalized.clone();

        let ones_matrix: DMatrix<f64> = DMatrix::from_element(input.shape().0, input.shape().1, 1.0);

        self.scaling_parameter * normalized + (ones_matrix * self.shifting_parameter)
    }
}

impl Layer for LayerNorm {
    fn calculate_gradients(&mut self, previous_gradient: DMatrix<f64>) -> DMatrix<f64> {
        const EPSILON_CONSTANT: f64 = 0.00001;
        
        for row in 0..previous_gradient.shape().0 {
            for col in 0..previous_gradient.shape().1 {
                self.scaling_gradient += self.prev_normalized_inputs[(row, col)] * previous_gradient[(row, col)];
                self.shifting_gradient += previous_gradient[(row, col)];
            }
        }

        let mut normalize_gradients: DMatrix<f64> = DMatrix::from_element(self.prev_normalized_inputs.shape().0, self.prev_normalized_inputs.shape().1, self.scaling_parameter);
        
        for row in 0..previous_gradient.shape().0 {
            for col in 0..previous_gradient.shape().1 {
                normalize_gradients[(row, col)] *= previous_gradient[(row, col)];
            }
        }
        
        // Calculating return gradients.
        let mut input_gradients: DMatrix<f64> = DMatrix::from_element(self.prev_inputs.shape().0, self.prev_inputs.shape().1, 0.0);
        let mut mean_gradients: DMatrix<f64> = DMatrix::from_element(1, self.prev_inputs.shape().1, 0.0);
        let mut variance_gradients: DMatrix<f64> = DMatrix::from_element(1, self.prev_inputs.shape().1, 0.0);

        for col in 0..variance_gradients.shape().1 {
            for row in 0..normalize_gradients.shape().0 {
                variance_gradients[(0, col)] += normalize_gradients[(row, col)] * (-1.0 * self.prev_variance[(0, col)].sqrt() * (self.prev_inputs[(row, col)] - self.prev_mean[(0, col)]) * (self.prev_variance[(0, col)] + EPSILON_CONSTANT).powf(-1.0 * (3.0 / 2.0)));
            }
        }

        for col in 0..mean_gradients.shape().1 {
            let mut column_total: f64 = 0.0;

            for row in 0..normalize_gradients.shape().0 {
                column_total += self.prev_inputs[(row, col)];
                mean_gradients[(0, col)] += normalize_gradients[(row, col)] * (-1.0 / (self.prev_variance[(0, col)] + EPSILON_CONSTANT).sqrt());
            }

            mean_gradients[(0, col)] += variance_gradients[(0, col)] * ((2.0 * self.prev_mean[(0, col)]) - ((2.0 * column_total) / (normalize_gradients.shape().0 as f64)))
        }

        for col in 0..variance_gradients.shape().1 {
            for row in 0..normalize_gradients.shape().0 {
                input_gradients[(row, col)] += normalize_gradients[(row, col)] * (1.0 / (self.prev_variance[(0, col)] + EPSILON_CONSTANT).sqrt());
                input_gradients[(row, col)] += variance_gradients[(0, col)] * ((2.0 * self.prev_inputs[(row, col)]) - (2.0 * self.prev_mean[(0, col)])) / normalize_gradients.shape().0 as f64;
                input_gradients[(row, col)] += mean_gradients[(0, col)] * (1.0 / normalize_gradients.shape().0 as f64);
            }
        }

        input_gradients
    }
    
    fn adjust_parameters(&mut self, learning_rate: f64) {
        self.scaling_parameter -= self.scaling_gradient * learning_rate;
        self.shifting_parameter -= self.shifting_gradient * learning_rate;
    }
}