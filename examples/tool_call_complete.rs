use ai_agent::{constant::GPT_OSS_120B_MODEL, llm::complete::chat_complete, tools::tools};
use anyhow::Ok;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv()?;

    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let tools = tools();

    let plan = chat_complete(
        GPT_OSS_120B_MODEL,
        Some("你是一个全能的助手"),
        "尼泊尔的首都是哪里？",
        tools.clone(),
    )
    .await?;

    tracing::info!("Response: {plan:#?}");
    println!("-----------------------------------------------");

    let plan = chat_complete(
        GPT_OSS_120B_MODEL,
        Some("你是一个全能的助手"),
        "5876乘以675是多少？",
        tools.clone(),
    )
    .await?;

    tracing::info!("Response: {plan:#?}");

    Ok(())
}
