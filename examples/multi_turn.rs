use ai_agent::{
    agent::Agent, 
    constant::GPT_4O_MINI_MODEL, tools::build_toolbox,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let toolbox = Arc::new(build_toolbox().await?);

    let agent = Agent::new(
        GPT_4O_MINI_MODEL,
        Some("你是一个全能的助手".to_string()),
        toolbox,
    );

    let result = agent
        .run(
            "我的名字叫杨旭，是一个软件开发工程师。",
            "multi_turn",
        )
        .await?;
    println!("\n回答1: {}", result.output);

    let result = agent
        .run(
            "我的叫什么名字，我的职业是什么？",
            "multi_turn",
        )
        .await?;
    println!("\n回答2: {}", result.output);

    Ok(())
}
