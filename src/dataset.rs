use csv;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray_csv::Array2Reader;
use std::collections::HashMap;
use std::str::FromStr;


pub struct Dataset {
    pub headers: Vec<String>,
    pub data: Array2<f64>,
    labels: Vec<i64>,
    label_names: HashMap<i64, String>,
}

impl Dataset {}

pub struct DatasetBuilder {
    unprocessed_data: Option<Array2<String>>,
    category_col: usize,
    select_columns: Vec<usize>,
    headers: Vec<String>,
}

impl DatasetBuilder {
    pub fn new() -> Self {
        DatasetBuilder {
            unprocessed_data: None,
            category_col: 0,
            select_columns: Vec::new(),
            headers: Vec::new(),
        }
    }

    pub fn read_csv(mut self, filename: &str) -> DatasetBuilder {
        let mut rdr = csv::Reader::from_path(filename).expect("failed to open csv file");
        self.unprocessed_data = Some(rdr.deserialize_array2_dynamic().unwrap());

        for element in rdr.headers().unwrap().into_iter() {
            self.headers.push(element.to_string());
        }
        self
    }

    pub fn select_columns(mut self, cols: &[usize]) -> DatasetBuilder {
        for i in cols {
            self.select_columns.push(*i);
        }
        self
    }

    pub fn class_column(mut self, col: usize) -> DatasetBuilder {
        self.category_col = col;
        self
    }

    pub fn build(self) -> Dataset {
        let unpr_data = match self.unprocessed_data {
            Some(x) => x,
            None => panic!("no dataset loaded"),
        };
        let mut features: Array2<f64> = Array2::<f64>::zeros((unpr_data.shape()[0], 0));

        for f in &self.select_columns {
            features = ndarray::concatenate![
                Axis(1),
                features,
                unpr_data
                    .column(*f)
                    .mapv(|elem| f64::from_str(&elem).unwrap())
                    .insert_axis(Axis(1))
            ];
        }
        let new_headers: Vec<String> = self
            .select_columns
            .iter()
            .map(|x| self.headers[*x].clone())
            .collect();

        let labels = unpr_data.column(self.category_col).to_vec();
        let unique_labels: Vec<&String> = labels.iter().unique().collect();
        let mut label_mapping: HashMap<i64, String> = HashMap::new();
        let mut inverse_mapping: HashMap<String, i64> = HashMap::new();
        for (index, element) in unique_labels.iter().enumerate() {
            label_mapping.insert(index as i64, (*element).clone());
            inverse_mapping.insert((*element).clone(), index as i64);
        }

        let labels: Vec<i64> = labels.into_iter().map(|x| inverse_mapping[&x]).collect();

        Dataset {
            data: features,
            labels: labels,
            label_names: label_mapping,
            headers: new_headers,
        }
    }
    //pub fn build(self) -> Dataset {}
}
