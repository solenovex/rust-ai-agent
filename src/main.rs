//! Multi-agent workflow demos. Pick a mode on the command line:
//!
//!   cargo run                          # same as `sequential`, default prompt
//!   cargo run -- sequential            # researcher -> writer -> editor
//!   cargo run -- sequential <prompt>   # same, with your own topic
//!   cargo run -- parallel              # optimist / pessimist / realist at once
//!   cargo run -- parallel <prompt>     # same, with your own idea to evaluate
//!   cargo run -- loop                  # writer <-> reviewer until score >= 85 (max 4 rounds)
//!   cargo run -- loop <prompt>         # same, with your own copywriting task
//!   cargo run -- composite             # researcher -> [3 analysts in parallel] -> [writer <-> reviewer loop] -> editor
//!   cargo run -- composite <prompt>    # same, with your own report topic
//!
//! The prompt is everything after the mode, so no quotes are needed:
//!
//!   cargo run -- parallel 评估我的 AI 教育平台创业想法
//!
//! sequential: only the researcher has tools (search etc.). Each agent sees
//!   the previous agent's work through the shared context.
//! parallel: three agents with no tools run concurrently on copies of the same
//!   context. Their new events are merged in agent order, and the output has
//!   one `[name]` section per agent.
//!
//! loop: a writer drafts, a reviewer scores it (structured output via the
//!   final_answer tool). The loop repeats until the score reaches PASS_SCORE or
//!   max iterations is hit. The final output is the reviewer's last Review.
//!
//! composite: all three workflows nested in one Sequential pipeline. From the
//!   outside it is still a single node. NOTE: this makes many LLM calls (research
//!   steps + 3 analysts + up to 3 write/review rounds + editor).
//!
//! All modes end by printing the result and the author of every event.

use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use study_agent::{
    agent::{
        context::ExecutionContext,
        model::{Agent, AgentResult, AgentStatus},
    },
    constant::{FINAL_ANSWER_TOOL_NAME, GPT_6_LUNA_PRO_MODEL},
    multi_agent::{
        node::AgentNode,
        workflow::{
            r#loop::LoopWorkflow, parallel::ParallelWorkflow, sequential::SequentialWorkflow,
        },
    },
    tool::{ToolBox, build_toolbox, final_answer::FinalAnswerTool},
};
use uuid::Uuid;

// ---------- shared helpers ----------

fn make_agent(name: &str, instructions: &str, toolbox: ToolBox, max_steps: usize) -> Agent {
    Agent::new(
        GPT_6_LUNA_PRO_MODEL.to_owned(),
        vec![instructions.to_owned()],
        Arc::new(toolbox),
    )
    .with_name(name.to_owned())
    .with_max_steps(max_steps)
}

/// Root context, built once by the caller (same as Agent::run does).
async fn new_root_context(agent: &Agent) -> anyhow::Result<ExecutionContext> {
    let session_id = Uuid::new_v4().to_string();
    let session = agent
        .session_manager
        .get_or_create(&session_id, None)
        .await?;
    Ok(ExecutionContext::new(session))
}

fn report(result: &AgentResult<serde_json::Value>) -> anyhow::Result<()> {
    match result.status {
        AgentStatus::Complete => {}
        AgentStatus::PendingConfirmation => anyhow::bail!(
            "a tool needs confirmation; workflows don't resume yet, remove that tool"
        ),
        status => anyhow::bail!("workflow stopped with status {status:?}"),
    }

    match &result.output {
        Some(serde_json::Value::String(text)) => println!("Result:\n{text}"),
        Some(other) => println!("Result:\n{}", serde_json::to_string_pretty(other)?),
        None => println!("Result: (no output)"),
    }

    println!("\nEvent authors:");
    for ev in &result.context.session.events {
        println!("- {}", ev.author);
    }
    Ok(())
}

// ---------- demo 1: sequential ----------

const RESEARCHER_INSTRUCTIONS: &str = "\
你是研究员。请用中文回答。
1. 使用搜索工具查找主题的最新资料,不要用相似关键词重复搜索同一个话题。
2. 最多使用 12 步;接近上限时,基于已有资料直接给出结果。
3. 只输出要点列表,并在每个要点后标注来源。";

/// composite 专用的研究员指令：限制搜索次数，要求收尾，并标注时效。
const COMPOSITE_RESEARCHER_INSTRUCTIONS: &str = "\
你是研究员。请用中文回答。当前时间是 2026 年 11 月。
1. 使用搜索工具查找主题的最新资料,优先使用最近一年的资料,不要用相似关键词重复搜索同一个话题。
2. 最多搜索 4 次;搜索完成后立即基于已有资料输出结果,不要再规划或反思。
3. 只输出要点列表,并在每个要点后标注来源链接(只使用搜索结果中出现过的 URL)。";

const WRITER_INSTRUCTIONS: &str = "\
你是写作者。请用中文,只基于前面研究员提供的要点和来源,写一篇 5 段左右的博客文章。
不要编造研究结果中没有的内容。只使用研究员提供的来源链接,不要自行补充 URL。";

const EDITOR_INSTRUCTIONS: &str = "\
你是编辑。请用中文审阅前面的文章,只修正语法并提升可读性。
不要新增内容,不要改变文章结构和长度,输出最终定稿。";

async fn run_sequential_demo(prompt: Option<String>) -> anyhow::Result<()> {
    let prompt = prompt.unwrap_or_else(|| "写一篇关于 AI Agent 最新趋势的文章".to_owned());

    // Only the researcher gets real tools (search etc.).
    let researcher_tools = build_toolbox().await?;
    tracing::info!(
        "researcher tools: {:?}",
        researcher_tools.keys().collect::<Vec<_>>()
    );

    let researcher = make_agent("researcher", RESEARCHER_INSTRUCTIONS, researcher_tools, 8);
    let writer = make_agent("writer", WRITER_INSTRUCTIONS, ToolBox::new(), 3);
    let editor = make_agent("editor", EDITOR_INSTRUCTIONS, ToolBox::new(), 3);

    let context = new_root_context(&researcher).await?;

    let workflow = SequentialWorkflow::new(vec![
        Arc::new(researcher) as Arc<dyn AgentNode>,
        Arc::new(writer),
        Arc::new(editor),
    ]);

    let result = workflow.execute(Some(prompt), context).await?;
    report(&result)
}

// ---------- demo 2: parallel ----------

const OPTIMIST_INSTRUCTIONS: &str = "\
你是乐观派分析师。请用中文,从积极的角度分析,聚焦机会和可能性。不超过 5 个要点。";

const PESSIMIST_INSTRUCTIONS: &str = "\
你是悲观派分析师。请用中文,聚焦风险和潜在问题进行分析。不超过 5 个要点。";

const REALIST_INSTRUCTIONS: &str = "\
你是务实派分析师。请用中文,从平衡的角度分析实际可行性。不超过 5 个要点。";

async fn run_parallel_demo(prompt: Option<String>) -> anyhow::Result<()> {
    let prompt = prompt.unwrap_or_else(|| "评估我的 AI 教育平台创业想法".to_owned());

    let optimist = make_agent("optimist", OPTIMIST_INSTRUCTIONS, ToolBox::new(), 3);
    let pessimist = make_agent("pessimist", PESSIMIST_INSTRUCTIONS, ToolBox::new(), 3);
    let realist = make_agent("realist", REALIST_INSTRUCTIONS, ToolBox::new(), 3);

    let context = new_root_context(&optimist).await?;

    let workflow = ParallelWorkflow::new(vec![
        Arc::new(optimist) as Arc<dyn AgentNode>,
        Arc::new(pessimist),
        Arc::new(realist),
    ]);

    let result = workflow.execute(Some(prompt), context).await?;
    report(&result)
}

// ---------- demo 3: loop ----------

/// What the reviewer returns through the final_answer tool.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Review {
    /// The exact text that was reviewed (the latest draft).
    content: String,
    /// Quality score from 0 to 100.
    score: u32,
    /// Concrete, actionable suggestions for improvement.
    feedback: String,
}

const PASS_SCORE: u32 = 85;

const LOOP_WRITER_INSTRUCTIONS: &str = "\
你是文案写作者。请用中文写营销文案。
如果前面的对话里已有评审员的反馈,请根据最新反馈改进上一版文案。只输出新的文案。";

const LOOP_REVIEWER_INSTRUCTIONS: &str = "\
你是严格的文案评审员。请用中文,评审写作者的最新一版文案,从清晰度、说服力、语法三方面打分(0-100,85 分以上才算优秀)。
content 字段填入你评审的文案原文;feedback 字段给出具体、可执行的改进建议。
content、score、feedback 三个字段必须全部填写,缺一不可;score 取清晰度、说服力、语法三项分数的平均值(四舍五入)。
必须调用 final_answer 返回结果。";

/// Stop when the reviewer's score reaches PASS_SCORE.
fn quality_reached(result: &AgentResult<serde_json::Value>, iteration: usize) -> bool {
    let Some(value) = &result.output else {
        return false;
    };
    match serde_json::from_value::<Review>(value.clone()) {
        Ok(review) => {
            tracing::info!("iteration {iteration}: score {}", review.score);
            review.score >= PASS_SCORE
        }
        Err(err) => {
            tracing::warn!("iteration {iteration}: could not parse Review: {err}");
            false
        }
    }
}

async fn run_loop_demo(prompt: Option<String>) -> anyhow::Result<()> {
    let prompt = prompt
        .unwrap_or_else(|| "为一款面向小学生的 AI 数学辅导 App 写一段营销文案".to_owned());

    let writer = make_agent("writer", LOOP_WRITER_INSTRUCTIONS, ToolBox::new(), 3);

    let mut reviewer_tools = ToolBox::new();
    reviewer_tools.insert(
        FINAL_ANSWER_TOOL_NAME.to_owned(),
        Box::new(FinalAnswerTool::<Review>::new()),
    );
    let reviewer = make_agent("reviewer", LOOP_REVIEWER_INSTRUCTIONS, reviewer_tools, 3);

    let context = new_root_context(&writer).await?;

    let workflow = LoopWorkflow::new(
        vec![Arc::new(writer) as Arc<dyn AgentNode>, Arc::new(reviewer)],
        4, // max iterations
    )
    .with_stop_condition(quality_reached);

    // Other ways to stop (swap in for `.with_stop_condition(quality_reached)`):
    //   keyword:  .with_stop_condition(|r, _| r.output.as_ref().is_some_and(|v| v.to_string().contains("APPROVED")))
    //   length:   .with_stop_condition(|r, i| i >= 3 && r.output.as_ref().map_or(0, |v| v.to_string().chars().count()) > 1000)

    let result = workflow.execute(Some(prompt), context).await?;
    report(&result)
}

// ---------- demo 4: composite ----------

fn analyst_instructions(role: &str, angle: &str) -> String {
    format!(
        "你是{role}分析师。请用中文,基于前面研究员提供的资料,{angle}。不超过 5 个要点。"
    )
}

const REPORT_WRITER_INSTRUCTIONS: &str = "\
你是报告写作者。请用中文,综合研究员的资料以及乐观、悲观、务实三方的分析,写一份 4-5 段的市场分析报告。
如果前面的对话里已有评审员的反馈,请根据最新反馈改进上一版报告。只使用研究员提供的来源链接,不要自行补充 URL。只输出报告正文。";

const REPORT_REVIEWER_INSTRUCTIONS: &str = "\
你是严格的报告评审员。请用中文,评审写作者的最新一版报告,从是否忠于资料、结构、清晰度三方面打分(0-100,85 分以上才算优秀)。
content 字段填入你评审的报告原文;feedback 字段给出具体、可执行的改进建议。
content、score、feedback 三个字段必须全部填写,缺一不可;score 取忠于资料、结构、清晰度三项分数的平均值(四舍五入)。
必须调用 final_answer 返回结果。";

const REPORT_EDITOR_INSTRUCTIONS: &str = "\
你是编辑。请用中文,审阅最新一版报告(评审员 content 字段中的正文),只修正语法并提升可读性。
不要新增内容,直接输出最终定稿。";

async fn run_composite_demo(prompt: Option<String>) -> anyhow::Result<()> {
    let prompt = prompt.unwrap_or_else(|| "写一份关于 AI Agent 市场的分析报告".to_owned());

    // Stage 1: research (the only agent with real tools).
    let mut researcher_tools = build_toolbox().await?;
    // 规划与反思工具会消耗步数，composite 的研究员只需要搜索。
    researcher_tools.remove("create_tasks");
    researcher_tools.remove("reflection");
    let researcher = make_agent("researcher", COMPOSITE_RESEARCHER_INSTRUCTIONS, researcher_tools, 12);

    // Stage 2: three perspectives at the same time.
    let optimist = make_agent(
        "optimist",
        &analyst_instructions("乐观派", "从积极的角度分析机会和可能性"),
        ToolBox::new(),
        3,
    );
    let pessimist = make_agent(
        "pessimist",
        &analyst_instructions("悲观派", "聚焦风险和潜在问题进行分析"),
        ToolBox::new(),
        3,
    );
    let realist = make_agent(
        "realist",
        &analyst_instructions("务实派", "从平衡的角度分析实际可行性"),
        ToolBox::new(),
        3,
    );
    let perspectives = ParallelWorkflow::new(vec![
        Arc::new(optimist) as Arc<dyn AgentNode>,
        Arc::new(pessimist),
        Arc::new(realist),
    ])
    .with_name("perspectives");

    // Stage 3: write / review loop.
    let writer = make_agent("writer", REPORT_WRITER_INSTRUCTIONS, ToolBox::new(), 3);
    let mut reviewer_tools = ToolBox::new();
    reviewer_tools.insert(
        FINAL_ANSWER_TOOL_NAME.to_owned(),
        Box::new(FinalAnswerTool::<Review>::new()),
    );
    let reviewer = make_agent("reviewer", REPORT_REVIEWER_INSTRUCTIONS, reviewer_tools, 3);
    let revision = LoopWorkflow::new(
        vec![Arc::new(writer) as Arc<dyn AgentNode>, Arc::new(reviewer)],
        3, // max iterations
    )
    .with_name("revision")
    .with_stop_condition(quality_reached);

    // Stage 4: final polish.
    let editor = make_agent("editor", REPORT_EDITOR_INSTRUCTIONS, ToolBox::new(), 3);

    let context = new_root_context(&researcher).await?;

    // Sequential -> Parallel -> Loop -> Agent, all behind the same trait.
    let workflow = SequentialWorkflow::new(vec![
        Arc::new(researcher) as Arc<dyn AgentNode>,
        Arc::new(perspectives),
        Arc::new(revision),
        Arc::new(editor),
    ])
    .with_name("report_pipeline");

    let result = workflow.execute(Some(prompt), context).await?;
    report(&result)
}

// ---------- entry ----------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    tracing_subscriber::fmt()
        .with_target(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "sequential".to_owned());
    let prompt = args.collect::<Vec<_>>().join(" ");
    let prompt = (!prompt.is_empty()).then_some(prompt);

    match mode.as_str() {
        "sequential" => run_sequential_demo(prompt).await,
        "parallel" => run_parallel_demo(prompt).await,
        "loop" => run_loop_demo(prompt).await,
        "composite" => run_composite_demo(prompt).await,
        other => anyhow::bail!(
            "unknown mode '{other}', use 'sequential', 'parallel', 'loop' or 'composite'"
        ),
    }
}