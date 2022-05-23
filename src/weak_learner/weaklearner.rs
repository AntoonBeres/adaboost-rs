use crate::dataset::Dataset;
use ndarray::prelude::*;

pub trait WeakLearner { 
    fn predict(&self, values: &Array2<f64>) -> Vec<i32>;
    fn get_alpha(&self) -> f64;
    fn set_alpha(&mut self, alpha: f64);
    fn train(weights: &Vec<f64>, data_set: &Dataset) -> Self;
}

