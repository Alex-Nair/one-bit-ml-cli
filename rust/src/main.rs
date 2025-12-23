pub mod one_bit_llm;
pub mod algorithms;

use nalgebra::*;

fn main() {
    let mut data: Vec<u8> = algorithms::train::load_tokens();
    let mut data_converted = algorithms::train::convert_to_usize(data.clone());
    let ohe_conversion: DMatrix<f64> = algorithms::train::one_hot_encoding(data_converted);
}