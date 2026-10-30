use std::sync::Arc;

use study_agent::{
    agent::model::Agent,
    constant::{FINAL_ANSWER_TOOL_NAME, GPT_6_LUNA_PRO_MODEL},
    llm::llm_answer::TextAnswer,
    tool::{build_toolbox, final_answer::FinalAnswerTool},
};
use uuid::Uuid;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    tracing_subscriber::fmt()
        .with_target(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    let mut toolbox = build_toolbox().await?;
    toolbox.insert(
        FINAL_ANSWER_TOOL_NAME.to_owned(),
        Box::new(FinalAnswerTool::<TextAnswer>::new()),
    );
    let agent = Agent::new(
        GPT_6_LUNA_PRO_MODEL.to_owned(),
        vec![
            "你是一个乐于助人的助手。遇到需要多个步骤的问题,必须先调用 create_tasks 制定计划,\
          每完成一个任务就再次调用 create_tasks 更新状态。只有任务真正完成才能标记为 completed,\
          工具调用失败时要重试或在回答中说明。请用中文回答。\
          只有拿到该任务的最终数值或结论之后,才能把它标记为 completed;\
          在所有计算都做完之前,不要把最后一个任务标记为 completed。"
                .to_owned(),
        ],
        Arc::new(toolbox),
    )
    .with_max_steps(15);

    let result = agent
        .run::<TextAnswer>(
            "我每月定投 3000 元,年化收益率 4%,按月复利计算,坚持 10 年后本金加收益一共有多少钱?\
         其中收益是多少?如果不投资、只是把钱存起来,10 年后比投资少多少?",
            &Uuid::new_v4().to_string(),
            None,
        )
        .await?;

    println!("Result: {result}");

    Ok(())
}
