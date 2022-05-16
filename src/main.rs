#![allow(dead_code)]
#![recursion_limit = "256"]
mod dataset;
mod stump;

use ndarray::Array2;
use stump::AdaboostModel;

//use ndarray::prelude::*;
fn main() {
    let cols: Vec<usize> = (2..32).collect();
    //2..32
    //

    let i = dataset::DatasetBuilder::new()
        .read_csv("datasets/breast-cancer.csv")
        .class_column(1)
        .select_columns(&cols[..]);

    /*let i = dataset::DatasetBuilder::new()
    .read_csv("datasets/diabetes.csv")
    .class_column(8)
    .select_columns(&cols[..]);*/

    let dset = i.build();

    let _k = Array2::from(vec![
        [6., 148., 72., 35., 0., 33.6, 0.627, 50.],
        [1., 85., 66., 29., 0., 26.6, 0.351, 31.],
    ]);

    let start = std::time::Instant::now();
    let _model = AdaboostModel::new(dset.get_n_features(), dset).fit();
    let end = start.elapsed();

    println!("time elapsed for fitting: {:?}", end);

    //let pred = model.get_prediction(&k);

    //model.pretty_print_prediction(&pred);

    //println!("datacols: {}", dset.get_data().ncols());
    //println!("labels: {}", dset.get_labels().len());

    //println!("{}",&dset.data);
    //println!("{:?}", &dset.get_headers());
}
