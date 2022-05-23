// writing this in rust was a mistake, help :(
//

#![allow(dead_code)]
use crate::dataset::Dataset;
use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use std::sync::{Arc, Mutex};
use crate::weak_learner::WeakLearner;

/// This determines the amount of decimals after the decimal point to be considered for choosing
/// unique tresholds, if you have a dataset with very small values (smaller, than 1e-4), consider
/// first scaling up your dataset by multiplying the datapoints
///
/// Higher training-speeds can be achieved by lowering the precision, can also be lowered to
/// prevent overfitting and getting a more general model
const PRECISION: f64 = 1e4;

/// A simple enum for representing the polarity of the stump
/// Used so that values are limited to +1 and -1
///i
/// Also could get optimized away by the compiler to use less memory than a full i32/i8 when using
/// lots of classifiers
enum Polarity {
    Positive,
    Negative,
}

impl Polarity {
    /// Get the polarity as an i32
    fn i32(&self) -> i32 {
        match self {
            Self::Positive => 1,
            Self::Negative => -1,
        }
    }
    /// Get the polarity as an f64
    fn f64(&self) -> f64 {
        match self {
            Self::Positive => 1.,
            Self::Negative => -1.,
        }
    }
    /// Get the polarity as a bool
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
    /// Weight of the weak learner for use in the adaboost model
    alpha: f64,
    /// The feature_id the stump uses for its predictions
    feature_id: usize,
}

impl Stump {
    fn new() -> Self {
        Stump {
            polarity: Polarity::Positive,
            treshold: None,
            alpha: 0.,
            feature_id: 0,
        }
    }
}

impl WeakLearner for Stump {
    /// Get a prediction from the stump
    fn predict(&self, values: &Array2<f64>) -> Vec<i32> {
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
    fn get_alpha(&self) -> f64 {
        self.alpha
    }
    fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }
    fn train(weights: &Vec<f64>, data_set: &Dataset) -> Self {
        let result = Self::new();
        let stump_mutex = Arc::new(Mutex::new(result));
        let lowest_err_mutex = Arc::new(Mutex::new(f64::INFINITY));
        let data = data_set.get_data();
        let labels = data_set.get_labels();
        data.axis_iter(Axis(1))
            .into_par_iter()
            .enumerate()
            .for_each(|(feature_id, data_col)| {
                let stump_copy = Arc::clone(&stump_mutex);

                //find all the unique values in the columns to use as potential tresholds
                let tholds: Vec<f64> = data_col
                    .iter()
                    .cloned()
                    //multiplication by 1000 and cast to int because floats are never really unique
                    .unique_by(|x| (*x * PRECISION) as i64)
                    .collect();

                //another 50% speedup on my machine by parallelizing this :) me=happy
                tholds.par_iter().for_each(|&t| {
                    let mut predictions: Vec<i32> = vec![1; labels.len()];
                    let mut p = Polarity::Positive;
                    predictions
                        .iter_mut()
                        .zip(data_col.iter())
                        .for_each(|(pred, d_point)| {
                            if *d_point < t {
                                *pred = -1
                            }
                        });
                    // Calculate the error by summing the weights of the misclassified samples
                    let mut error: f64 = labels
                        .iter()
                        .zip(predictions.iter())
                        .zip(weights.iter())
                        .map(|((label, pred), w)| if *pred != *label { *w } else { 0. })
                        .sum();
                    // If the error is greater than 0.5, invert the weak learning stump, by
                    // inverting the polarity
                    if error > 0.5 {
                        error = 1. - error;
                        p = Polarity::Negative;
                    }
                    // If this error is smaller than the previously smallest error, update the
                    // stump classifier
                    let mut lowest_err = lowest_err_mutex.lock().unwrap();
                    if error < *lowest_err {
                        *lowest_err = error;
                        //drop the lock as soon as possible so other threads can continue,
                        //slight micro-optimization..
                        std::mem::drop(lowest_err);
                        let mut stump = stump_copy.lock().unwrap();
                        stump.polarity = p;
                        stump.treshold = Some(t);
                        stump.feature_id = feature_id;
                    }
                });
            });

        {
            let mut stump = stump_mutex.lock().unwrap();
            //guards so that the lowest_err mutex-lock gets dropped after it is no longer needed
            let lowest_err = lowest_err_mutex.lock().unwrap();
            stump.alpha = 0.5
                * ((1.0 - *lowest_err + f64::MIN_POSITIVE) / (*lowest_err + f64::MIN_POSITIVE))
                    .ln();
            std::mem::drop(stump)
        }

        Arc::try_unwrap(stump_mutex)
            .unwrap_or(Mutex::new(Stump::new()))
            .into_inner()
            .unwrap()
    }
}


