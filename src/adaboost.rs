use crate::dataset::Dataset;
use crate::weak_learner::Stump;
use crate::weak_learner::WeakLearner;
use crate::modelsample::*;
use ndarray::parallel::prelude::*;



/// The actual adaboost model class
pub struct AdaboostModel {
    /// The classifiers, which are weak-learning stumps
    classifiers: Vec<Stump>,
    /// The dataset loaded into the model
    dataset: Dataset,
    n_classifiers: usize,
}

impl AdaboostModel {
    /// Create a new model from a idataset with a specified amount of weak classifiers
    /// # Arguments
    /// * `n_classifiers` - the amount of weak classifiers to be used
    /// * `dataset` - the `Dataset` to be loaded into the model, the model takes ownership of the
    /// dataset
    pub fn new(n_classifiers: usize, dataset: Dataset) -> Self {
        AdaboostModel {
            classifiers: Vec::new(),
            dataset,
            n_classifiers,
        }
    }
    /// Get a prediction from the trained model
    /// # Arguments
    /// * `samples_dyn` - An array of samples in a format that implements the ModelSample trait
    pub fn get_prediction(&self, samples_dyn: &dyn ModelSample) -> Vec<i32> {
        let m_samples = samples_dyn.to_adamodel_sample();
        let samples = m_samples.get();
        if samples.ncols() != self.dataset.get_n_features() {
            panic!("amount of features in sample doesnt correspond to amount of features in model");
        }
        let predicts: Vec<Vec<f64>> = self
            .classifiers
            .par_iter()
            .map(|s| {
                s.predict(samples)
                    .iter_mut()
                    .map(|res| *res as f64 * s.get_alpha())
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
                    .get(&(*x.1 as i32))
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
        for _i in 0..self.n_classifiers {
            let stump = Stump::train(&weights, &self.dataset);
            let preds = stump.predict(data);

            // Update the weights of the samples
            let mut sum_of_weights = 0.;
            weights
                .iter_mut()
                .zip(labels.iter())
                .zip(preds.iter())
                .for_each(|((w, lbl), p)| {
                    *w *= (-stump.get_alpha() * *lbl as f64 * *p as f64).exp();
                    sum_of_weights += *w;
                });
            //renormalize the weights again
            weights.iter_mut().for_each(|w| {
                *w /= sum_of_weights;
            });
            self.classifiers.push(stump);
        }

        // free the dataset from memory after training is done, this is done for memory-efficiency
        // reasons
        self.dataset.free_dset_memory();
    }
}
