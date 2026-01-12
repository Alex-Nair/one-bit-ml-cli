use std::fmt::Debug;

use std::ops::{Add, Sub, Mul, Div};
use num_traits::{Num, Pow};

pub trait Numeric: Num + Clone + Debug + Default + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Pow<Self, Output = Self> {}

impl<T> Numeric for T
where 
    T: Num + Clone + Debug + Default + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Pow<Self, Output = Self>,
{

}

#[derive(Clone)]
pub struct Matrix<T: Numeric> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T> // 1D vector used instead of Vec<Vec<f32>> to maximise space efficiency.
}


impl<T: Numeric> Matrix<T> {
    pub fn new(rows: usize, cols: usize, init_val: T) -> Matrix<T> {
        Matrix {
            rows: rows,
            cols: cols,
            data: vec![init_val; rows * cols]
        }
    }

    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[col + row * self.cols].clone()
    }

    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.data[col + row * self.cols] = value;
    }

    pub fn display(&self) {
        println!("Rows: {}, Cols: {}", self.rows, self.cols);
        println!("Data: {:?}", self.data);
    }


    // Raising to the power (unit-wise)
    pub fn pow_unit(&mut self, pow: T) -> Matrix<T> {
        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone().pow(pow.clone()));
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }

    pub fn pow_matrix(&mut self, other: Matrix<T>) -> Matrix<T> {
        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone().pow(other.data[i].clone()));
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }
}


impl <T: Numeric> Add for Matrix<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            println!("ERROR ENCOUNTERED - Adding two matrices.");
            println!("Dimension mismatch: ({}, {}) + ({}, {})", self.rows, self.cols, other.rows, other.cols);
            return self; // Fallback option.
        }

        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone() + other.data[i].clone());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }
}


impl <T: Numeric> Sub for Matrix<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            println!("ERROR ENCOUNTERED - Subtracting two matrices.");
            println!("Dimension mismatch: ({}, {}) + ({}, {})", self.rows, self.cols, other.rows, other.cols);
            return self; // Fallback option.
        }

        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone() - other.data[i].clone());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }
}


impl <T: Numeric> Mul for Matrix<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.cols != other.rows {
            println!("ERROR ENCOUNTERED - Multiplying two matrices.");
            println!("Dimension mismatch: ({}, {}) * ({}, {})", self.rows, self.cols, other.rows, other.cols);
            return self; // Fallback option.
        }

        let mut data: Vec<T> = vec![];

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut total: T = T::default();

                for k in 0..self.cols {
                    total = total + self.get(i, k) * other.get(k, j);
                }

                data.push(total);
            }
        }

        Matrix {
            rows: self.rows,
            cols: other.cols,
            data: data
        }
    }
}


impl <T: Numeric> Div for Matrix<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            println!("ERROR ENCOUNTERED - Dividing two matrices.");
            println!("Dimension mismatch: ({}, {}) + ({}, {})", self.rows, self.cols, other.rows, other.cols);
            return self; // Fallback option.
        }

        let mut data: Vec<T> = vec![];

        for i in 0..self.data.len() {
            data.push(self.data[i].clone() / other.data[i].clone());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data
        }
    }
}