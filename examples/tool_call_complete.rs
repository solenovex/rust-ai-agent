use ai_agent::{constant::GPT_4O_MINI_MODEL, llm::complete::chat_complete, tools::tools};
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

    let result = chat_complete(
        GPT_4O_MINI_MODEL,
        Some(r#"你是一个全能的助手。今天的日期是2026年7月22日。
你可以使用工具来搜索最新信息。
重要：当工具返回搜索结果时，你必须直接使用这些结果来回答，不要说"信息尚未公布"或"我不知道"。
你的训练数据有截止日期，可能已经过时，请始终优先信任工具返回的内容。"#),
        "2026世界杯决赛的比分是？",
        tools.clone(),
    )
    .await?;

    tracing::info!("Response: {result:#?}");

    Ok(())
}
