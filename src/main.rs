use ai_agent::{constant::GPT_OSS_120B_MODEL, llm::{structured::chat_complete_structured}};
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

    let plan = chat_complete_structured(
        GPT_OSS_120B_MODEL,
        Some("你是一个全能的助手"),
        "我要去美加墨世界杯观看比赛，如果安排？",
    )
    .await?;

    println!("Response: {plan:#?}");

    Ok(())
}
