use itertools::Itertools;
use ndarray::prelude::*;
use ndarray_csv::Array2Reader;
use std::collections::HashMap;
use std::str::FromStr;

/// A Dataset object for easy use in machine-learning algorithms
pub struct Dataset {
    /// The headers, describing the columns in the dataset
    pub headers: Vec<String>,
    /// The actual data in the dataset
    pub data: Array2<f64>,
    /// The labels of the dataset in the form integers
    labels: Vec<i64>,
    /// A mapping of label-id's to their actual names
    label_names: HashMap<i64, String>,
}

impl Dataset {
    /// Get the amount of features for fitting present in the dataset
    pub fn get_n_features(&self) -> usize {
        self.headers.len()
    }
    /// Get an immutable reference to the data of the dataset
    pub fn get_data(&self) -> &Array2<f64> {
        &self.data
    }
    /// Get an immutable reference to the data-labels
    pub fn get_labels(&self) -> &Vec<i64> {
        &self.labels
    }
    /// Get the mapping of label-id's (integers) to corresponding names
    pub fn get_label_mapping(&self) -> &HashMap<i64, String> {
        &self.label_names
    }

    pub fn get_headers(&self) -> &Vec<String> {
        &self.headers
    } 
}
/**
 * DatasetBuilder: builder-pattern, struct to load and prepare a dataset
 */
pub struct DatasetBuilder {
    unprocessed_data: Option<Array2<String>>,
    category_col: usize,
    select_columns: Vec<usize>,
    headers: Vec<String>,
}

impl DatasetBuilder {
    /// Create a new empty DatasetBuilder
    pub fn new() -> Self {
        DatasetBuilder {
            unprocessed_data: None,
            category_col: 0,
            select_columns: Vec::new(),
            headers: Vec::new(),
        }
    }
    /// Load a csv-file into the DatasetBuilder
    /// # Arguments
    /// * `filename` - the path to the .csv file
    pub fn read_csv(mut self, filename: &str) -> DatasetBuilder {
        let mut rdr = csv::Reader::from_path(filename).expect("failed to open csv file");
        self.unprocessed_data = Some(rdr.deserialize_array2_dynamic().unwrap());

        for element in rdr.headers().unwrap().into_iter() {
            self.headers.push(element.to_string());
        }
        self
    }
    /// select the columns to be used for fitting the data
    /// # Arguments
    /// * `cols` - slice with column indexes that will be loaded into the dataset
    pub fn select_columns(mut self, cols: &[usize]) -> DatasetBuilder {
        for i in cols {
            self.select_columns.push(*i);
        }
        self
    }
    /// Select the column with the dataset labels
    /// # Arguments
    /// * `col` - The index of the column that contains the labels
    pub fn class_column(mut self, col: usize) -> DatasetBuilder {
        self.category_col = col;
        self
    }
    /// Build the dataset
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
            labels,
            label_names: label_mapping,
            headers: new_headers,
        }
    }
    //pub fn build(self) -> Dataset {}
}
