use crate::gaia::{
    models::{GaiaEvalResult, GaiaRow},
    solver::{GAIA_PROMPT, solve_problem_with_retry},
};

fn is_correct(prediction: &str, answer: &str) -> bool {
    if prediction.is_empty() {
        false
    } else {
        prediction.trim().to_lowercase() == answer.trim().to_lowercase()
    }
}

pub async fn evaluate_gaia_single(problem: GaiaRow, model: &str) -> GaiaEvalResult {
    let result = solve_problem_with_retry(model, GAIA_PROMPT, &problem.question).await;
    match result {
        Ok(output) => GaiaEvalResult {
            task_id: problem.task_id,
            model: String::from(model),
            correct: is_correct(&output.final_answer, &problem.final_answer),
            is_solvable: Some(output.is_solvable),
            prediction: Some(output.final_answer),
            answer: problem.final_answer,
            unsolvable_reason: Some(output.unsolvable_reason),
            error: None,
        },
        Err(err) => GaiaEvalResult {
            task_id: problem.task_id,
            model: String::from(model),
            correct: false,
            is_solvable: None,
            prediction: None,
            answer: problem.final_answer,
            unsolvable_reason: None,
            error: Some(err.to_string()),
        },
    }
}
