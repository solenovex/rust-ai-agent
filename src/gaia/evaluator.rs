use std::sync::Arc;

use crate::{
    gaia::{
        model::{GaiaEvalResult, GaiaOutput, GaiaRow},
        solver::{solve_gaia_problem_with_retry, solve_gaia_problem_with_tools_with_retry},
    },
    tool::ToolBox,
};

pub const GAIA_PROMPT: &str = r#"You are a general AI assistant. I will ask you a question.
First, determine if you can solve this problem with your current capabilities and set "is_solvable" accordingly.
If you can solve it, set "is_solvable" to true and provide your answer in "final_answer".
If you cannot solve it, set "is_solvable" to false and explain why in "unsolvable_reason".
Your final answer should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.
If you are asked for a number, don't use a comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g., for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma-separated list, apply the above rules depending on whether the element is a number or a string."#;

fn is_correct(prediction: &str, answer: &str) -> bool {
    if prediction.is_empty() {
        false
    } else {
        prediction.trim().to_lowercase() == answer.trim().to_lowercase()
    }
}

fn to_eval_result(
    problem: GaiaRow,
    model: &str,
    result: anyhow::Result<GaiaOutput>,
) -> GaiaEvalResult {
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

pub async fn evaluate_gaia_single(problem: GaiaRow, model: &str) -> GaiaEvalResult {
    let result = solve_gaia_problem_with_retry(model, GAIA_PROMPT, &problem.question).await;
    to_eval_result(problem, model, result)
}

pub async fn evaluate_gaia_single_with_tools(
    problem: GaiaRow,
    model: &str,
    toolbox: Arc<ToolBox>,
) -> GaiaEvalResult {
    let result =
        solve_gaia_problem_with_tools_with_retry(model, GAIA_PROMPT, &problem, toolbox).await;
    to_eval_result(problem, model, result)
}
