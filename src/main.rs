#![allow(dead_code)]
mod dataset;
mod stump;

//use ndarray::prelude::*;
fn main() {
    let cols: Vec<usize> = (2..32).collect();

    let i = dataset::DatasetBuilder::new()
        .read_csv("datasets/breast-cancer.csv")
        .class_column(1)
        .select_columns(&cols[..]);
    let dset = i.build();
    let u = dset.data.row(0);
    let i = u;

    for i in dset.data.rows() {
        println!("{}", i);
    }
    //println!("{}",&dset.data);
    println!("{:?}", &dset.headers);
}
