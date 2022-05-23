// writing this in rust was a mistake, help :(
//

#![allow(dead_code)]
use crate::dataset::Dataset;
use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use std::fmt::Debug;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

/// This determines the amount of decimals after the decimal point to be considered for choosing
/// unique tresholds, if you have a dataset with very small values (smaller, than 1e-4), consider
/// first scaling up your dataset by multiplying the datapoints
///
/// Higher training-speeds can be achieved by lowering the precision, can also be lowered to
/// prevent overfitting and getting a more general model
const PRECISION: f64 = 1e4;

/// A simple enum for representing the polarity of the stump
/// Used so that values are limited to +1 and -1
///
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

trait WeakLearner {
    fn predict(&self, values: &Array2<f64>) -> Vec<i32>;
    fn get_alpha(&self) -> f64;
    fn set_alpha(&mut self, alpha: f64);

    fn train(weights: Vec<f64>, data_set: &Dataset) -> Self;
}

/// A weak-learning stump, essentially a binary tree with 2 branches and height 1
/// Through boosting, many weak learners together perform like a strong learner
///
/// # TLDR
/// Stump alone weak. Stumps together strong
struct Stump {
    /// Treshold of the stump
    treshold: Option<f64>,
    /// Polarity of the stump, -1 or +1
    polarity: Polarity,
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

/*impl Debug for Stump {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        
    }
}*/

impl WeakLearner for Stump {
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
    fn train(weights: Vec<f64>, data_set: &Dataset) -> Self {
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
                    let mut predictions: Vec<i64> = vec![1; labels.len()];
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
        }
        let result = Arc::try_unwrap(stump_mutex).unwrap_or(panic!("error")).into_inner().unwrap();
        result
        //Self::new() //TODO
    }
}

/// The actual adaboost model class
pub struct AdaboostModel {
    /// The classifiers, which are weak-learning stumps
    classifiers: Vec<Stump>,
    /// The dataset loaded into the model
    dataset: Dataset,
}

pub enum ModelSampleType<'a> {
    Reference(&'a Array2<f64>),
    Move(Array2<f64>),
}

impl<'a> ModelSampleType<'a> {
    fn get(&self) -> &Array2<f64> {
        match self {
            ModelSampleType::Reference(x) => x,
            ModelSampleType::Move(x) => x,
        }
    }
}

pub trait ModelSample {
    fn to_adamodel_sample(&self) -> ModelSampleType;
}

impl ModelSample for Vec<String> {
    fn to_adamodel_sample(&self) -> ModelSampleType {
        let samplesize = self.len();
        let vecf64 = self
            .iter()
            .map(|s| f64::from_str(s.as_str()).unwrap())
            .collect();
        let i: Array2<f64> = Array2::from_shape_vec((1, samplesize), vecf64).unwrap();
        ModelSampleType::Move(i)
    }
}

impl ModelSample for Vec<f64> {
    fn to_adamodel_sample(&self) -> ModelSampleType {
        let samplesize = self.len();
        ModelSampleType::Move(Array2::from_shape_vec((1, samplesize), self.to_vec()).unwrap())
    }
}

impl ModelSample for Array2<f64> {
    fn to_adamodel_sample(&self) -> ModelSampleType {
        ModelSampleType::Reference(self)
    }
}

impl ModelSample for Array2<String> {
    fn to_adamodel_sample(&self) -> ModelSampleType {
        let mut result: Array2<f64> = Array2::<f64>::zeros((self.shape()[0], 0));

        //this could be done more efficiently.. but will do for now
        for f in 0..self.ncols() {
            result = ndarray::concatenate![
                Axis(1),
                result,
                self.column(f)
                    .mapv(|elem| f64::from_str(&elem).unwrap())
                    .insert_axis(Axis(1))
            ];
        }
        ModelSampleType::Move(result)
    }
}

impl ModelSample for [Vec<String>] {
    fn to_adamodel_sample(&self) -> ModelSampleType {
        let length = self.len();
        let samplesize = self.len();
        let mut result_vec: Vec<f64> = Vec::new();
        for i in self {
            for j in i {
                result_vec.push(f64::from_str(j).unwrap());
            }
        }
        let i: Array2<f64> = Array2::from_shape_vec((length, samplesize), result_vec).unwrap();
        ModelSampleType::Move(i)
    }
}

impl ModelSample for Dataset {
    fn to_adamodel_sample(&self) -> ModelSampleType {
        ModelSampleType::Reference(self.get_data())
    }
}

impl AdaboostModel {
    /// Create a new model from a dataset with a specified amount of weak classifiers
    /// # Arguments
    /// * `n_classifiers` - the amount of weak classifiers to be used
    /// * `dataset` - the `Dataset` to be loaded into the model, the model takes ownership of the
    /// dataset
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
    /// Get a prediction from the trained model
    pub fn get_prediction(&self, samples_dyn: &ModelSampleType) -> Vec<i32> {
        let samples = samples_dyn.get();
        if samples.ncols() != self.dataset.get_n_features() {
            panic!("amount of features in sample doesnt correspond to amount of features in model");
        }
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
    /// The fitting function, here the magic happens!
    /// This function takes ownership of the model and returns the fitted model, this way of
    /// working is necessary for parallelism
    ///
    /// This function heavily relies on parallelism for optimal speed (through rayon via
    /// ndarray::parallel)
    pub fn fit(&mut self) {
        let n_samples: usize = self.dataset.get_data().nrows();
        let labels = self.dataset.get_labels();

        // check if the amount of labels corresponds to the amount of samples
        if labels.len() != n_samples {
            panic!("labels-size and samplesize not the same");
        }

        let data = self.dataset.get_data();

        //Initially all samples have equal weight, weights are normalized
        let mut weights: Vec<f64> = vec![1.0 / (n_samples as f64); n_samples];

        //this one sadly can't be parallelized, because the adaboost algorithm depends on the order
        //in which the weak classifiers are trained..
        self.classifiers.iter_mut().for_each(|stump_i| {
            // Create some reference-counted mutexes for fancy thread-safe concurrency
            let lowest_err_mutex = Arc::new(Mutex::new(f64::INFINITY));
            let stump_mutex = Arc::new(Mutex::new(stump_i));

            //Here the fun begins, a concurrent for-loop over all the columns, takes only a single
            //column from the dataset (single feature) and loops over the values concurrentls, this
            //is to find the best feature for the classifier
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
                        let mut predictions: Vec<i64> = vec![1; labels.len()];
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

            let mut stump = stump_mutex.lock().unwrap();
            //guards so that the lowest_err mutex-lock gets dropped after it is no longer needed
            {
                let lowest_err = lowest_err_mutex.lock().unwrap();
                stump.alpha = 0.5
                    * ((1.0 - *lowest_err + f64::MIN_POSITIVE) / (*lowest_err + f64::MIN_POSITIVE))
                        .ln();
            }

            // here the old code ends
            //
            //


            

            //predictions of this stump
            let preds = stump.predict(data);

            // Update the weights of the samples
            let mut sum_of_weights = 0.;
            weights
                .iter_mut()
                .zip(labels.iter())
                .zip(preds.iter())
                .for_each(|((w, y), p)| {
                    *w *= (-stump.alpha * *y as f64 * *p as f64).exp();
                    sum_of_weights += *w;
                });
            //renormalize the weights again
            weights.iter_mut().for_each(|w| {
                *w /= sum_of_weights;
            });
        });

        // free the dataset from memory after training is done, this is done for memory-efficiency
        // reasons
        self.dataset.free_dset_memory();
    }
}
