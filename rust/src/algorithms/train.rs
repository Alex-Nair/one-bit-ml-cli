use std::{fs::File, io::{Read, Seek, SeekFrom}};
use nalgebra::*;

pub fn load_tokens() -> Vec<u8> {
    // Commented for testing purposes. The below code block only extracts the first 16 MB of data.
    // let data = fs::read("../../dataset/tokens.bin").expect("Failed to read file.");

    let mut input_file = File::open("../../dataset/tokens.bin").expect("Failed to read file.");
    let mut data: Vec<u8> = vec![0; 16000];
    let _ = input_file.seek(SeekFrom::Start(0));
    let _ = input_file.read_exact(&mut data);

    data
}

pub fn convert_to_usize(input: Vec<u8>) -> Vec<usize> {
    let mut processed_data: Vec<usize> = vec![];

    for chunk in input.chunks_exact(4) {
        processed_data.push((chunk[0] as usize) * 256^3 + (chunk[1] as usize) * 256^2 + (chunk[2] as usize) * 256 + (chunk[3] as usize));
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