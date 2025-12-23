use std::fs;
use nalgebra::*;

pub fn load_tokens() -> Vec<u8> {
    let data = fs::read("../../dataset/tokens.bin").expect("Failed to read file.");

    let mut sample: Vec<u8> = vec![];

    // For debugging purposes: Shrink the dataset for faster testing.
    for i in 0..1000 {
        sample.push(data[i]);
    }

    sample
}

pub fn convert_to_usize(input: Vec<u8>) -> Vec<usize> {
    let mut processed_data: Vec<usize> = vec![];

    for chunk in input.chunks_exact(4) {
        processed_data.push(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize);
    }

    for element in processed_data.clone() {
        print!("{}", element)
    }

    processed_data
}

pub fn one_hot_encoding(data: Vec<usize>) -> DMatrix<f64> {
    let mut result: DMatrix<f64> = DMatrix::from_element(*data.iter().max().unwrap() + 1, data.len(), 0.0);

    for (index, element) in data.iter().enumerate() {
        result[(*element, index)] = 1.0;
    }

    result
}