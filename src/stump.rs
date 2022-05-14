#![allow(dead_code)]
use crate::dataset::Dataset;
use itertools::Itertools;
use ndarray::prelude::*;
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
            feature_id: 0,
        }
    }

    pub fn predict(&self, values: &Array2<f64>) -> Vec<i32> {
        let tres = self.treshold.unwrap();

        let filt = |i: &f64| -> i32 {
            if *i < tres {
                -1 * self.polarity
            } else {
                1 * self.polarity
            }
        };
        values
            .column(self.feature_id)
            .into_iter()
            .map(filt)
            .collect()
    }
}

pub struct AdaboostModel {
    classifiers: Vec<Stump>,
    dataset: Dataset,
}

impl AdaboostModel {
    pub fn new(n_classifiers: usize, dataset: Dataset) -> Self {
        let mut result = AdaboostModel {
            classifiers: Vec::new(),
            dataset,
        };
        for _i in 0..n_classifiers {
            result.classifiers.push(Stump::new());
        }
        result
    }
    pub fn get_prediction(&self, samples: &Array2<f64>) -> Vec<i32> {
        let n_samples: usize = samples.ncols();
        let mut result_all: Vec<Vec<i32>> = vec![Vec::new(); self.classifiers.len()];

        let mut result = vec![0; n_samples];

        self.classifiers.iter().enumerate().for_each(|clf_pair| {
            result_all[clf_pair.0] = clf_pair.1.predict(samples);
        });

        result_all.iter().for_each(|v| {
            v.iter().enumerate().for_each(|val_pair| {
                result[val_pair.0] += val_pair.1;
            });
        });

        result
    }

    pub fn pretty_print_prediction(&self, prediction: &Vec<i32>) {
        prediction.iter().enumerate().for_each(|x| {
            println!(
                "sample {}: {}",
                x.0,
                self.dataset
                    .get_label_mapping()
                    .get(&(*x.1 as i64))
                    .unwrap()
            );
        });
    }

    pub fn fit(&mut self) {
        let features_count = self.dataset.get_n_features();
        let n_samples: usize = self.dataset.get_data().ncols();

        let labels = self.dataset.get_labels();
        if (labels.len() != n_samples) {
            panic!("labels-size and samplesize not the same");
        }

        let data = self.dataset.get_data();

        let mut weights: Vec<f64> = vec![(1.0 / (n_samples as f64)); n_samples];

        for stump in &mut self.classifiers {
            let mut lowest_err = f64::INFINITY;

            for feature_id in 0..features_count {
                let data_col = data.column(feature_id).to_vec();
                let tholds: Vec<f64> = data_col
                    .into_iter()
                    .unique_by(|x| (x * 1000.) as i64)
                    .collect();

                let mut predictions: Vec<i32> = vec![1; n_samples];
                tholds.iter().for_each(|t| {
                    let p = 1;

                    data_col.iter().enumerate().for_each(|data_p_pair| {
                        if (data_p_pair.1 < t) {
                            predictions[data_p_pair.0] = -1
                        }
                    });

                    let error: f64 = labels
                        .iter()
                        .zip(predictions.iter())
                        .zip(weights.iter())
                        .map(
                            |((label, pred), w)| {
                                if (*pred as i64 != *label) {
                                    *w
                                } else {
                                    0.
                                }
                            },
                        )
                        .sum();

                    if error > 0.5 {
                        error = 1. - error;
                        p = -1;
                    }
                    if error < lowest_err {
                        stump.polarity = p;
                        stump.treshold = Some(*t);
                        stump.feature_id = feature_id;
                        lowest_err = error;
                    }
                });
            }
        }

        self.dataset.free_dset_memory();

        //for tuple in samplesX {}
    }
}
