extern crate csv;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate kaggle_core;
extern crate rand;

use std::error::Error;
use std::fs::File;
use std::rc::Rc;

use rand::{thread_rng, Rng};

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

#[derive(Debug, Deserialize, Clone)]
pub struct TestPassangerDTO {
    #[serde(rename = "PassengerId")]
    passenger_id: String,
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

#[derive(Debug, Serialize, Clone)]
pub struct ResultDTO {
    #[serde(rename = "PassengerId")]
    passenger_id: String,
    #[serde(rename = "Survived")]
    survived: u8,
}

make_kd_tree!(3, u8, titanic_kd_tree);
type DataSet = Vec<(titanic_kd_tree::Vector, Rc<titanic_kd_tree::TValue>)>;
type TestSet = Vec<(titanic_kd_tree::Vector, String)>;

fn read_training_data() -> Result<DataSet, Box<Error>> {
    let file = File::open("data/train.csv").unwrap();
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .flexible(true)
        .from_reader(file);
    let mut result: Vec<TrainingPassangerDTO> = vec![];
    for record in reader.deserialize() {
        result.push(record?);
    }
    Ok(result
        .iter()
        .map(|p| {
            let sex = if p.sex.contains("female") { 1.0 } else { -1.0 };
            let age = match p.age {
                Some(a) => a,
                None => 25.0,
            };
            (
                [p.pclass as f32 * 1.5, sex * 5.0, age * 0.35],
                Rc::new(p.survived),
            )
        })
        .collect())
}

fn read_test_data() -> Result<TestSet, Box<Error>> {
    let file = File::open("data/test.csv").unwrap();
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .flexible(true)
        .from_reader(file);
    let mut result: Vec<TestPassangerDTO> = vec![];
    for record in reader.deserialize() {
        result.push(record?);
    }
    Ok(result
        .iter()
        .map(|p| {
            let sex = if p.sex.contains("female") { 1.0 } else { -1.0 };
            let age = match p.age {
                Some(a) => a,
                None => 25.0,
            };
            (
                [p.pclass as f32 * 1.5, sex * 5.0, age * 0.35],
                p.passenger_id.clone(),
            )
        })
        .collect())
}

fn run_test(split_n: usize) -> i32 {
    let mut data: DataSet = read_training_data().unwrap();

    thread_rng().shuffle(&mut data);
    let (test, training) = data.split_at_mut(split_n);

    let tree = titanic_kd_tree::new(training).unwrap();

    let result: i32 = test
        .iter()
        .map(|passanger| {
            let nearest = tree.find_k_nearest(passanger.0, 5);
            let result: f32 = nearest.iter().map(|x| *x.2 as f32).sum();
            let result = result / 5.0;
            let actual = *passanger.1;
            let result = if result < 0.5 { 0 } else { 1 };
            if result == actual {
                1
            } else {
                0
            }
        })
        .sum();

    let perc = result as f32 / split_n as f32 * 100 as f32;
    println!("Final score: {} out of {} [{}%]", result, split_n, perc);
    result
}

fn run_actual() {
    let mut data: DataSet = read_training_data().unwrap();
    let tree = titanic_kd_tree::new(&mut data).unwrap();

    let test_data = read_test_data().unwrap();
    let result = test_data.iter().map(|passanger| {
        let nearest = tree.find_k_nearest(passanger.0, 5);
        let result: f32 = nearest.iter().map(|x| *x.2 as f32).sum();
        let result = result / 5.0;
        let result = if result < 0.5 { 0 } else { 1 };
        ResultDTO {
            passenger_id: passanger.1.clone(),
            survived: result,
        }
    });

    let file = File::create("data/result.csv").unwrap();
    let mut writer = csv::Writer::from_writer(file);

    result.for_each(|p| {
        writer.serialize(p).unwrap();
    });
    writer.flush().unwrap();
}

fn main() {
    for _ in 0..20 {
        run_test(100);
    }
    run_actual();
}
