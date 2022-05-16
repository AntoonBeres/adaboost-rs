// writing this in rust was a mistake, help :(
// 

#![allow(dead_code)]
use crate::dataset::Dataset;
use itertools::Itertools;
use ndarray::prelude::*;

/// A simple enum for representing the polarity of the stump 
/// Used so that values are limited to +1 and -1
///
/// Also could get optimized away by the compiler to use less memory than a full i32/i8 when using
/// lots classifiers
enum Polarity {
    Positive,
    Negative,
}

impl Polarity {
    fn i32(&self) -> i32 {
        match self {
            Self::Positive => 1,
            Self::Negative => -1,
        }
    }
    fn f64(&self) -> f64 {
        match self {
            Self::Positive => 1.,
            Self::Negative => -1.,
        }
    }
    fn bool(&self) -> bool {
        match self {
            Self::Positive => true,
            Self::Negative => false,
        }
    }
}





/// A weak-learning stump, essentially a binary tree with 2 branches and height 1
/// Through boosting, many weak learners together perform like a strong learner
///
/// # TLDR
/// Stump alone weak. Stumps together strong
pub struct Stump {
    /// Treshold of the stump
    treshold: Option<f64>,
    /// Polarity of the stump, -1 or +1
    polarity: Polarity,
    alpha: f64,
    /// The feature_id the stump uses for its predictions
    feature_id: usize,
}

impl Stump {
    pub fn new() -> Self {
        Stump {
            polarity: Polarity::Positive,
            treshold: None,
            alpha: 0.,
            feature_id: 0,
        }
    }

    pub fn predict(&self, values: &Array2<f64>) -> Vec<i32> {
        let tres = self.treshold.unwrap();
        values
            .column(self.feature_id)
            .iter()
            .map(|x| match x {
                x if *x < tres => -self.polarity.i32(),
                x if *x >= tres => self.polarity.i32(),
                _ => panic!("this should never happen, is x nan?"),
            })
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
        let predicts: Vec<Vec<f64>> = self
            .classifiers
            .iter()
            .map(|s| {
                s.predict(samples)
                    .iter_mut()
                    .map(|res| *res as f64 * s.alpha)
                    .collect()
            })
            .collect();

        let mut y_pred = vec![0.; samples.nrows()];

        for i in &predicts[..] {
            y_pred.iter_mut().zip(i.iter()).for_each(|(y, p)| *y += *p);
        }

        let result = y_pred
            .iter()
            .map(|x| match x {
                x if *x >= 0.0 => 1,
                x if *x < 0.0 => -1,
                _ => panic!("random unexpected stuff"),
            })
            .collect();

        result
    }

    pub fn pretty_print_prediction(&self, prediction: &[i32]) {
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
        let n_samples: usize = self.dataset.get_data().nrows();

        let labels = self.dataset.get_labels();
        if labels.len() != n_samples {
            panic!("labels-size and samplesize not the same");
        }

        let data = self.dataset.get_data();

        let mut weights: Vec<f64> = vec![1.0 / (n_samples as f64); n_samples];

        for stump in &mut self.classifiers {
            let mut lowest_err = f64::INFINITY;

            for feature_id in 0..features_count {
                let data_col = data.column(feature_id).to_vec();
                let tholds: Vec<f64> = data_col
                    .iter()
                    .cloned()
                    .unique_by(|x| (*x * 1000.) as i64)
                    .collect();

                tholds.iter().for_each(|t| {
                    let mut predictions: Vec<i64> = vec![1; labels.len()];
                    let mut p = Polarity::Positive;

                    predictions
                        .iter_mut()
                        .zip(data_col.iter())
                        .for_each(|(pred, d_point)| {
                            if *d_point < *t {
                                *pred = -1
                            }
                        });

                    // Calculate the error
                    let mut error: f64 = labels
                        .iter()
                        .zip(predictions.iter())
                        .zip(weights.iter())
                        .map(
                            |((label, pred), w)| {
                                if *pred as i64 != *label {
                                    *w
                                } else {
                                    0.
                                }
                            },
                        )
                        .sum();

                    // If the error is greater than 0.5, invert the weak learning stump, by
                    // inverting the polarity
                    if error > 0.5 {
                        error = 1. - error;
                        p = Polarity::Negative;
                    }
                    // If this error is smaller than the previously smallest error, update the
                    // stump classifier
                    if error < lowest_err {
                        stump.polarity = p;
                        stump.treshold = Some(*t);
                        stump.feature_id = feature_id;
                        lowest_err = error;
                    }
                });
            }

            stump.alpha = 0.5
                * ((1.0 - lowest_err + f64::MIN_POSITIVE) / (lowest_err + f64::MIN_POSITIVE)).ln();
            let preds = stump.predict(data);

            // Update the weights of the samples
            weights
                .iter_mut()
                .zip(labels.iter())
                .zip(preds.iter())
                .for_each(|((w, y), p)| {
                    *w *= (-stump.alpha * *y as f64 * *p as f64).exp();
                });

            let sum_of_weights: f64 = weights.iter().sum();
            weights.iter_mut().for_each(|w| {
                *w /= sum_of_weights;
            });
        }

        // free the dataset from memory after training is done, this is done for memory-efficiency
        // reasons
        self.dataset.free_dset_memory();
    }
}
