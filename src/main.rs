use std::sync::Arc;

use study_agent::{
    agent::model::Agent,
    constant::{FINAL_ANSWER_TOOL_NAME, GPT_6_LUNA_PRO_MODEL},
    llm::llm_answer::TextAnswer,
    tool::{build_toolbox, final_answer::FinalAnswerTool},
};
use uuid::Uuid;

const INSTRUCTIONS: &str = "\
你是一个研究助手。请用中文回答。

研究流程:
1. 研究任何主题时,第一步必须先调用 get_wikipedia_page 查询维基百科背景资料。
2. 如果 get_wikipedia_page 调用失败,必须先调用 reflection 分析失败原因和替代方案,再改用 web_search。不要重复调用同一个失败的工具。
3. 每完成 2-3 次搜索,必须调用 reflection,总结已经找到什么、还缺什么、是否已经足够回答。
4. 不要用相似的关键词重复搜索同一个主题。如果某个主题已经有可靠来源,就换下一个主题。
5. 如果不同来源的数据冲突,调用 reflection 说明你选择哪个数据以及理由。
6. 在调用 final_answer 之前,必须先调用 reflection 做一次自检:确认没有遗漏重要内容。
7. 总共最多使用 8 步。接近上限时,即使信息不完整,也要基于已有内容给出最终回答,并说明哪些方面没有覆盖。";

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

    tracing::info!("tools: {:?}", toolbox.keys().collect::<Vec<_>>());

    let agent = Agent::new(
        GPT_6_LUNA_PRO_MODEL.to_owned(),
        vec![INSTRUCTIONS.to_owned()],
        Arc::new(toolbox),
    )
    .with_max_steps(10);

    let result = agent
        .run::<TextAnswer>(
            "研究量子计算领域近期的重要进展,重点关注纠错方面,给出一份简明的总结。",
            &Uuid::new_v4().to_string(),
            None,
        )
        .await?;

    println!("Result: {result}");

    Ok(())
}