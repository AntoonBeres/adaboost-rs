# adaboost-rs
A simple implementation of adaboost in Rust using (concurrent) decision stumps as weak learners 
## Compilation
compile the library with:  
`cargo build --release`  
## Running examples
first compile the example eg:  
`cargo build --release --example breast-cancer`  
then run:  
`./target/release/examples/breast-cancer`


Or just instantly run the example without storing the compiled binary:  
`cargo run --release --example breast-cancer`

