use std::sync::Arc;

use study_agent::{
    constant::{self, FINAL_ANSWER_TOOL_NAME},
    gaia::{
        dataset::{download_gaia_dataset_attachments, load_gaia_dataset},
        evaluator::{evaluate_gaia_single, evaluate_gaia_single_with_tools},
        model::{GaiaEvalResult, GaiaOutput, GaiaRow},
    },
    llm::semaphore::get_semaphore,
    tool::{build_toolbox, final_answer::FinalAnswerTool},
};
use tokio::task::JoinSet;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    tracing_subscriber::fmt()
        .with_target(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    gaia_experiment().await
}

pub async fn gaia_experiment() -> anyhow::Result<()> {
    let model = constant::GPT_6_LUNA_PRO_MODEL;
    let problems = load_gaia_dataset().await?;
    download_gaia_dataset_attachments().await?;

    let problems_with_files: Vec<&GaiaRow> =
        problems.iter().filter(|r| r.file_name.is_some()).collect();
    let problems_with_zips: Vec<&GaiaRow> = problems
        .iter()
        .filter(|r| r.file_name.clone().is_some_and(|r| r.ends_with(".zip")))
        .collect();

    println!("Total problems: {}", problems.len());
    println!("Problem with attachments: {}", problems_with_files.len());
    println!("Problem with zips: {}", problems_with_zips.len());

    let mut set = JoinSet::new();

    for problem in problems.iter() {
        let problem = problem.clone();
        set.spawn(async move {
            let _permit = get_semaphore().acquire().await?;
            let eval = evaluate_gaia_single(problem, model).await;
            // drop(permit);
            Ok::<_, anyhow::Error>(eval)
        });
    }

    let mut results_without_tools: Vec<GaiaEvalResult> = Vec::new();
    let mut failed_without_tools = 0usize;
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(Ok(eval)) => {
                tracing::info!("{eval:#?}");
                if !eval.correct {
                    failed_without_tools += 1;
                }
                results_without_tools.push(eval);
            }
            Ok(Err(e)) => {
                failed_without_tools += 1;
                tracing::error!("Task error: {e:#}");
            }
            Err(join_err) => {
                failed_without_tools += 1;
                tracing::error!("Task panicked/cancelled: {join_err}")
            }
        }
    }

    let correct_without_tools = results_without_tools.iter().filter(|r| r.correct).count();

    // With Tools

    let mut toolbox = build_toolbox().await?;
    toolbox.insert(
        FINAL_ANSWER_TOOL_NAME.to_owned(),
        Box::new(FinalAnswerTool::<GaiaOutput>::new()),
    );
    let toolbox = Arc::new(toolbox);

    let mut set = JoinSet::new();

    for problem in problems.iter() {
        let problem = problem.clone();
        let toolbox = toolbox.clone();
        set.spawn(async move {
            let _permit = get_semaphore().acquire().await?;
            let eval = evaluate_gaia_single_with_tools(problem, model, toolbox).await;
            // drop(permit);
            Ok::<_, anyhow::Error>(eval)
        });
    }

    let mut results_with_tools: Vec<GaiaEvalResult> = Vec::new();
    let mut failed_with_tools = 0usize;
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(Ok(eval)) => {
                tracing::info!("{eval:#?}");
                if !eval.correct {
                    failed_with_tools += 1;
                }
                results_with_tools.push(eval);
            }
            Ok(Err(e)) => {
                failed_with_tools += 1;
                tracing::error!("Task error: {e:#}");
            }
            Err(join_err) => {
                failed_with_tools += 1;
                tracing::error!("Task panicked/cancelled: {join_err}")
            }
        }
    }

    let correct_with_tools = results_with_tools.iter().filter(|r| r.correct).count();

    println!(
        "Without tools done: {} evaluated, {} failed, accuracy = {:.2}%",
        results_without_tools.len(),
        failed_without_tools,
        correct_without_tools as f64 / results_without_tools.len().max(1) as f64 * 100.0
    );

    println!(
        "With tools done: {} evaluated, {} failed, accuracy = {:.2}%",
        results_with_tools.len(),
        failed_with_tools,
        correct_with_tools as f64 / results_with_tools.len().max(1) as f64 * 100.0
    );

    Ok(())
}
