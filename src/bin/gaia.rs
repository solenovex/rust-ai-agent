use std::collections::HashMap;

use ai_agent::{
    gaia::{dataset::load_gaia_level1, evaluator::evaluate_gaia_single, models::GaiaEvalResult},
    llm::semaphore::get_semaphore,
};
use tokio::task::JoinSet;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    gaia_level1_experiment().await
}

const MODELS: &[&str] = &["openai/gpt-4.1-mini", "anthropic/claude-haiku-4.5"];

pub async fn gaia_level1_experiment() -> anyhow::Result<()> {
    let problems = load_gaia_level1().await?;
    let mut set = JoinSet::new();
    for model in MODELS.iter() {
        for problem in problems.iter() {
            let problem = problem.clone();
            set.spawn(async move {
                let permit = get_semaphore().acquire().await?;
                let eval = evaluate_gaia_single(problem, model).await;
                drop(permit);
                Ok::<_, anyhow::Error>(eval)
            });
        }
    }

    let mut results: HashMap<String, Vec<GaiaEvalResult>> = HashMap::new();
    while let Some(Ok(result)) = set.join_next().await {
        match result {
            Ok(eval) => {
                tracing::info!("{eval:#?}");
                results.entry(eval.model.clone()).or_default().push(eval);
            }
            Err(e) => tracing::error!("task panicked: {e}"),
        }
    }

    for (model, evals) in &results {
        let correct = evals.iter().filter(|e| e.correct).count();
        let total = evals.len();
        tracing::info!(
            "{model}: {correct}/{total} ({:.1}%)",
            correct as f64 / total as f64 * 100.0
        );
    }

    Ok(())
}
