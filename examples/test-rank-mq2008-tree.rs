extern crate gbdt;
use std::convert::TryFrom;
use std::thread;
use std::time::Duration;

use gbdt::decision_tree::PredVec;
use gbdt::gradient_boost::GBDT;
use gbdt::input;

fn main() {
    // // Call this command to convert xgboost model:
    // // python examples/convert_xgboost.py xgb-data/xgb_rank_pairwise/xgb.model "rank:pairwise" xgb-data/xgb_rank_pairwise/gbdt.model
    // // load model
    // let gbdt = GBDT::from_xgoost_dump("examples/my_model.json", "rank:pairwise")
    //     .expect("failed to load model");

    // // load test data
    // let test_file = "xgb-data/xgb_rank_pairwise/mq2008.test";
    // let mut input_format = input::InputFormat::txt_format();
    // input_format.set_feature_size(47);
    // input_format.set_delimeter(' ');
    // let ori_test_data = input::load(test_file, input_format).expect("failed to load test data");

    let mut handles = Vec::new();

    let mut total = Duration::new(0, 0);

    let num_query = 30;

    for _x in 0..num_query {
        let handle = thread::spawn(move || {
            // load model
            let gbdt = GBDT::from_xgoost_dump("examples/my_model.json", "rank:pairwise")
                .expect("failed to load model");

            // load test data
            let test_file = "xgb-data/xgb_rank_pairwise/mq2008.test";
            let mut input_format = input::InputFormat::txt_format();
            input_format.set_feature_size(47);
            input_format.set_delimeter(' ');
            let test_data = input::load(test_file, input_format).expect("failed to load test data");

            // inference
            println!("start prediction");
            use std::time::Instant;
            let now = Instant::now();
            let predicted: PredVec = gbdt.predict(&test_data);
            let elapsed = now.elapsed();
            println!("Elapsed: {:.2?}", elapsed);
            let data_size = u32::try_from(test_data.len());

            println!("Per Query: {:.2?}", elapsed / data_size.unwrap());

            assert_eq!(predicted.len(), test_data.len());
            println!("{}", test_data.len());
            println!("prediction Done");
            elapsed
        });
        handles.push(handle);
    }

    for handle in handles {
        let per_q = handle.join().unwrap();
        total += per_q;
    }

    println!("number of query: {}", num_query);
    println!("average per query: {:.2?}", total / num_query);
    println!("average per doc: {:.2?}", total / num_query / 200);
}
