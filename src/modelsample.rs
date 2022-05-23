use ndarray::prelude::*;
use crate::dataset::Dataset;
use std::str::FromStr;



pub enum ModelSampleType<'a> {
    Reference(&'a Array2<f64>),
    Move(Array2<f64>),
}

impl<'a> ModelSampleType<'a> {
    pub fn get(&self) -> &Array2<f64> {
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

