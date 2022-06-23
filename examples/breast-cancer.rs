use adaboost_rs::AdaboostModel;
use adaboost_rs::dataset;

//new branch
//use ndarray::prelude::*;
fn main() {
    let cols: Vec<usize> = (2..32).collect();
    //2..32
    //
    let dset_validation = dataset::DatasetBuilder::new()
        .read_csv("datasets/breast-cancer-training.csv")
        .class_column(1)
        .select_columns(&cols[..]);

    let dset_test = dataset::DatasetBuilder::new()
        .read_csv("datasets/breast-cancer-test.csv")
        .class_column(1)
        .select_columns(&cols[..]);
    

    let dset_validation = dset_validation.build();
    let dset_test = dset_test.build();

    let start = std::time::Instant::now();
    let mut model = AdaboostModel::new(30, dset_validation);
    model.fit();
    let end = start.elapsed();

    println!("time elapsed for fitting: {:?}", end);
    

    let test_labels = dset_test.get_labels();
    

    let test_prediction = model.get_prediction(&dset_test);



    let mut total_correct: i32 = 0;

    test_prediction.iter().zip(test_labels.iter()).for_each(|(pred, label)| {
        if pred == label {
            total_correct += 1;
        }
    });
    
    let correct_ratio: f64 = total_correct as f64 / test_labels.len() as f64;


    println!("percentage correctly classified: {:.2}%", correct_ratio*100.);
    println!("error rate: {:.2}%", (1.0-correct_ratio)*100.);

}
