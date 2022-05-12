#![allow(dead_code)]
use std::cmp::Ordering;
use std::collections::HashMap;
use ndarray::prelude::*;
use crate::dataset::Dataset;


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

    pub fn predict(&self, values: &Array1<f64>) -> Vec<i32> {
        let tres = self.treshold.unwrap();

        let filt = |i: &f64| -> i32 {
            if *i < tres {
                -1
            } else {
                1
            }
        };
        let predictions: Vec<i32> = values.into_iter().map(filt).collect();
        predictions
    }
}


pub struct AdaboostModel {
    classifiers: Vec<Stump>,
}

impl AdaboostModel {
    pub fn new(n_classifiers: usize) -> Self {
        let mut result = AdaboostModel {
            classifiers: Vec::new(),
        };
        for _i in 0..n_classifiers {
            result.classifiers.push(Stump::new());
        }
        result
    }
    pub fn get_prediction(&self, sample: &[f64]) -> i32 {
        0
    }

    pub fn fit(&mut self, dataset: &Dataset) {
        let features_count = dataset.get_n_features();
        let labels = dataset.get_labels();
        let data = dataset.get_data();


        for stump in &mut self.classifiers {
            let mut lowest_err = f64::INFINITY;

            for feature_id in 0..features_count {
                let data_col = data.column(feature_id);
            }
            
        }
        

        //for tuple in samplesX {}
    }
}
