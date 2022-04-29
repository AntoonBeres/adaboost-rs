#![allow(dead_code)]
use std::cmp::Ordering;


pub struct Stump<T: Ord> {
    pub value: T,
    pub prediction: Option<i64>,
    treshold: T,
    polarity: i32,
    alpha: f64,
}

impl<T: Ord> Stump<T>{
    pub fn predict(&self, values: Vec<T>) -> Vec<i32>{
        let length = values.len();
        let predictions = vec![1; length];
        
        let filt = |i: T| -> i32 {
            if i < self.treshold {-1} else {1}
        };
        let predictions: Vec<i32> = values.into_iter().map(filt).collect();
        return predictions;

    }


}


pub struct Adaboost_Model{
    n_classifiers: usize,

}

impl Adaboost_Model{
    pub fn get_prediction(&self, sample: &Vec<f64>) -> i32 {
        0
    }
    pub fn fit(&mut self, samplesX: &Vec<Vec<f64>>, labelsY: &Vec<i32>){

    }
    

}



