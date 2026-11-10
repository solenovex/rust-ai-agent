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
//!   cargo run -- agent-tool            # 协调员把 researcher / coder 当作工具调用
//!   cargo run -- agent-tool <prompt>   # 同上，使用自定义的调研与编码任务
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
    tool::{
        AgentTool, GithubStarsTool, ToolBox, build_toolbox, code::register_code_tools,
        final_answer::FinalAnswerTool, web_search::WebSearchTool,
    },
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

// ---------- 演示 5: Agent as Tool ----------

const AGENT_TOOL_RESEARCHER_INSTRUCTIONS: &str = "\
你是研究专员。请用中文回答。
工作方式:
1. 先调用 github_stars 工具,一次查询任务中提到的所有 GitHub 仓库,repos 参数传 \"owner/repo\" 列表。
2. 只有需要补充框架简介或确认仓库归属时,才使用 web_search,最多 2 次。
3. 最终只返回一个 Markdown 表格,列为:框架 | GitHub 仓库 | Star 数 | 来源 URL。
   - Star 数必须使用 github_stars 返回的精确值,不要使用搜索结果摘要、排行榜或旧讨论帖里的数字。
   - 来源 URL 使用 github_stars 返回的仓库 URL。
   - 某个仓库查询失败时,Star 数写“查询失败”,并在来源 URL 写错误原因。
4. 不要输出搜索过程,也不要编造数据。";

const AGENT_TOOL_CODER_INSTRUCTIONS: &str = "\
你是 Python 专家。请用中文回答。
调用方会在 request 中提供一份调研表格(框架、GitHub 仓库、Star 数)。
严格要求:
1. 只使用 request 中给出的数据,把它们硬编码到脚本里。
2. 禁止访问网络、禁止调用 GitHub API、禁止安装第三方库,只使用 Python 标准库。
3. 脚本输出 Markdown 表格和 ASCII 柱状图。
4. 执行脚本,返回一次完整代码和真实运行输出;不要重复解释同一段代码。
5. 如果执行失败,返回错误信息,不要编造运行结果。";

const AGENT_TOOL_COORDINATOR_INSTRUCTIONS: &str = "\
你是项目协调员。请用中文回答。
工作流程固定如下:
1. 先调用 researcher,获取一份“框架 | GitHub 仓库 | Star 数 | 来源 URL”的表格。
2. 再调用 coder,把 researcher 返回的完整表格原样放进 request,要求它用这些数据生成 Markdown 表格、ASCII 柱状图和可运行脚本。
3. 最终回答只包含以下内容,不要重复 coder 的整段解释:
   - 最终 Markdown 表格
   - ASCII 柱状图
   - 一份完整 Python 脚本(只出现一次)
   - 一句关于数据来源和 Star 数会变化的说明
注意:
- coder 不联网,所有数字必须来自 researcher。
- 不要自己编造 Star 数或 URL。";

/// 演示 Agent as Tool: coordinator 的 toolbox 里放的是另外两个 Agent。
///
/// 与 workflow 的关键区别:子代理在独立 session 中运行,coordinator 只看到
/// 它们的最终输出,看不到中间的工具调用和推理过程。
async fn run_agent_tool_demo(prompt: Option<String>) -> anyhow::Result<()> {
    let prompt = prompt.unwrap_or_else(|| {
        "调研最新的 AI Agent 框架(如 LangGraph、CrewAI、AutoGen),\
         并写一个 Python 脚本比较它们的 GitHub star 数"
            .to_owned()
    });

    // 1) 研究专员:github_stars 精确取数,web_search 只补充背景。
    let mut researcher_tools = ToolBox::new();
    researcher_tools.insert("github_stars".to_owned(), Box::new(GithubStarsTool));
    researcher_tools.insert("web_search".to_owned(), Box::new(WebSearchTool));
    let researcher = Arc::new(
        make_agent(
            "researcher",
            AGENT_TOOL_RESEARCHER_INSTRUCTIONS,
            researcher_tools,
            6,
        )
        .with_description(
            "联网搜索并总结任意主题的最新资料。适合需要事实、来源或最新信息的任务。",
        ),
    );

    // 2) 编码专员:拥有沙箱代码执行能力(需配置 CODE_EXECUTION)。
    let mut coder_tools = ToolBox::new();
    let sandbox_prompt = register_code_tools(&mut coder_tools).await?;
    if coder_tools.is_empty() {
        tracing::warn!(
            "code execution is not configured; the coder agent can only write code, not run it"
        );
    }
    let mut coder_instructions = AGENT_TOOL_CODER_INSTRUCTIONS.to_owned();
    if !sandbox_prompt.is_empty() {
        coder_instructions.push_str("\n\n");
        coder_instructions.push_str(&sandbox_prompt);
    }
    let coder = Arc::new(
        make_agent("coder", &coder_instructions, coder_tools, 8)
            .with_description(
                "编写并执行 Python 代码,用于数据分析和可视化。适合计算、统计或生成图表的任务。",
            ),
    );

    // 3) 协调员:普通 toolbox,只是把两个 Agent 包装成 Tool。
    //    子代理内部需要确认的工具(如 execute_python)由 AgentTool 自动批准。
    let mut orchestrator_tools = ToolBox::new();
    orchestrator_tools.insert(
        "researcher".to_owned(),
        Box::new(AgentTool::new(researcher)),
    );
    orchestrator_tools.insert(
        "coder".to_owned(),
        Box::new(AgentTool::new(coder).with_auto_approve_child_tools(true)),
    );

    let orchestrator = make_agent(
        "coordinator",
        AGENT_TOOL_COORDINATOR_INSTRUCTIONS,
        orchestrator_tools,
        8,
    );

    let context = new_root_context(&orchestrator).await?;
    // Agent 本身实现了 AgentNode,所以可以直接作为单节点执行。
    let result = orchestrator.execute(Some(prompt), context).await?;
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
        "agent-tool" => run_agent_tool_demo(prompt).await,
        other => anyhow::bail!(
            "unknown mode '{other}', use 'sequential', 'parallel', 'loop', 'composite' or 'agent-tool'"
        ),
    }
}