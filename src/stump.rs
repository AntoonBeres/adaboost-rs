#![allow(dead_code)]
use std::cmp::Ordering;
use std::collections::HashMap;

pub struct Stump {
    //pub value: T,
    //pub prediction: Option<i64>,
    treshold: Option<f64>,
    polarity: i32,
    alpha: f64,
    feature_id: usize,
}

impl Stump {
    pub fn new() -> Self {
        Stump {
            polarity: 1,
            treshold: None,
            alpha: 0.,
            feature_id: 0
        }
    }

    pub fn predict(&self, values: Vec<f64>) -> Vec<i32> {
        let tres = self.treshold.unwrap();

        let filt = |i: f64| -> i32 {
            if i < tres {
                -1
            } else {
                1
            }
        };
        let predictions: Vec<i32> = values.into_iter().map(filt).collect();
        predictions
    }
}



pub struct DatasetTuple {
    datafeatures: Vec<f64>,
    features_map: HashMap<usize, String>,

}


pub struct Adaboost_Model {
    classifiers: Vec<Stump>,
}

impl Adaboost_Model {
    pub fn new(n_classifiers: usize) -> Self {
        let mut result = Adaboost_Model {
            classifiers: Vec::new(),
        };
        for i in 0..n_classifiers {
            result.classifiers.push(Stump::new());
        }
        result
    }
    pub fn get_prediction(&self, sample: &Vec<f64>) -> i32 {
        0
    }

    pub fn fit(&mut self, samplesX: &Vec<Vec<f64>>, labelsY: &Vec<i32>) {
        let n_features = samplesX[0].len();
        for stump in &mut self.classifiers {
            let mut lowest_err = f64::INFINITY;

            for feature in 0..n_features {
                
            }
            
        }

        for tuple in samplesX {}
    }
}
