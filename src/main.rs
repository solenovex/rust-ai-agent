use std::sync::Arc;

use study_agent::{
    agent::{
        model::{Agent, AgentResult, AgentStatus},
        tool_confirmation::ToolConfirmation,
    },
    constant::{FINAL_ANSWER_TOOL_NAME, GPT_6_LUNA_PRO_MODEL},
    llm::llm_answer::TextAnswer,
    tool::{build_toolbox, final_answer::FinalAnswerTool},
};
use uuid::Uuid;

const INSTRUCTIONS: &str = "\
你是一个研究助手。请用中文回答。

研究流程:
1. 每完成 2-3 次搜索,必须调用 reflection,总结已经找到什么、还缺什么、是否已经足够回答。
2. 不要用相似的关键词重复搜索同一个主题。如果某个主题已经有可靠来源,就换下一个主题。
3. 如果工具调用失败,先调用 reflection 分析原因,再决定换哪个工具。
4. 如果不同来源的数据冲突,调用 reflection 说明你选择哪个数据以及理由。
5. 在调用 final_answer 之前,必须先调用 reflection 做一次自检:确认没有遗漏重要内容。
6. 总共最多使用 8 步。接近上限时,即使信息不完整,也要基于已有内容给出最终回答,并说明哪些方面没有覆盖。";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    tracing_subscriber::fmt()
        .with_target(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    let mut toolbox = build_toolbox().await?;
    let sandbox_prompt = study_agent::tool::code::register_code_tools(&mut toolbox).await?;

    toolbox.insert(
        FINAL_ANSWER_TOOL_NAME.to_owned(),
        Box::new(FinalAnswerTool::<TextAnswer>::new()),
    );

    tracing::info!("tools: {:?}", toolbox.keys().collect::<Vec<_>>());

    let mut instructions = vec![INSTRUCTIONS.to_owned()];
    if !sandbox_prompt.is_empty() {
        instructions.push(sandbox_prompt);
    }

    let agent = Agent::new(
        GPT_6_LUNA_PRO_MODEL.to_owned(),
        instructions,
        Arc::new(toolbox),
    )
    .with_max_steps(10);

    let args: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with("--")).collect();
    let prompt = if args.is_empty() {
        "请在沙箱中用 Python 计算 1 到 100 的平方和,再用 bash 命令验证该结果,最后给出结论。".to_owned()
    } else {
        args.join(" ")
    };
    let session_id = Uuid::new_v4().to_string();

    // Auto-approve any tool calls that require confirmation (e.g. code execution
    // tools), so the demo runs end to end without manual interaction.
    let mut confirmations: Option<Vec<ToolConfirmation>> = None;
    let mut result: AgentResult<TextAnswer>;
    loop {
        result = agent
            .run::<TextAnswer>(&prompt, &session_id, confirmations.take())
            .await?;
        if result.status == AgentStatus::Complete {
            break;
        }
        if result.status == AgentStatus::PendingConfirmation {
            let confs: Vec<ToolConfirmation> = result
                .context
                .session
                .pending_tool_calls
                .iter()
                .map(|pc| ToolConfirmation {
                    tool_call_id: pc.tool_call.id.clone(),
                    reason: None,
                    approved: true,
                    modified_arguments: None,
                })
                .collect();
            confirmations = Some(confs);
            continue;
        }
        anyhow::bail!("agent stopped with status {:?}", result.status);
    }

    println!("Result: {result}");

    Ok(())
}