extern crate csv;
#[macro_use]
extern crate serde_derive;

#[macro_use]
mod kdtree;

use std::error::Error;
use std::fs::File;
use std::rc::Rc;

#[derive(Debug, Deserialize, Clone)]
pub struct TrainingPassangerDTO {
    #[serde(rename = "PassengerId")]
    passenger_id: String,
    #[serde(rename = "Survived")]
    survived: u8,
    #[serde(rename = "Pclass")]
    pclass: u8,
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Sex")]
    sex: String,
    #[serde(rename = "Age")]
    age: Option<f32>,
    #[serde(rename = "SibSp")]
    sib_sp: Option<usize>,
    #[serde(rename = "Parch")]
    parch: Option<usize>,
    #[serde(rename = "Ticket")]
    ticket: Option<String>,
    #[serde(rename = "Fare")]
    fare: Option<f32>,
    #[serde(rename = "Cabin")]
    cabin: Option<String>,
    #[serde(rename = "Embarked")]
    embarked: Option<char>,
}

fn read_training_data() -> Result<Vec<TrainingPassangerDTO>, Box<Error>> {
    let file = File::open("data/train.csv").unwrap();
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .flexible(true)
        .from_reader(file);
    let mut result = vec![];
    for record in reader.deserialize() {
        // println!("{:#?}", record);
        result.push(record?);
    }
    Ok(result)
}

make_kd_tree!(2, TrainingPassangerDTO, titanic_kd_tree);

fn main() {
    let result = read_training_data().unwrap();

    let tree = titanic_kd_tree::new(vec![
        ([1.0, 2.0], Rc::new(result[0].clone())),
        ([1.0, 2.0], Rc::new(result[1].clone())),
        ([1.0, 2.0], Rc::new(result[2].clone())),
        ([1.0, 2.0], Rc::new(result[3].clone())),
        ([1.0, 2.0], Rc::new(result[4].clone())),
        ([1.0, 2.0], Rc::new(result[5].clone())),
        ([1.0, 2.0], Rc::new(result[6].clone())),
        ([1.0, 2.0], Rc::new(result[7].clone())),
    ]).unwrap();

    let result = tree.find_k_nearest([2.3, 5.8], 4);

    println!("Done! {} {:#?}", result.len(), tree);
}
