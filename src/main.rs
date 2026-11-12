//! 终端交互式 AI Agent。
//!
//! 这是一个行式 REPL，不进入 alternate screen：
//! - 正常使用终端滚动和复制粘贴；
//! - 助手回答用 Markdown 渲染；
//! - 支持 transfer、工具审批、长期记忆、沙箱、MCP 和上下文优化。

use std::{
    collections::HashMap,
    io::{self, IsTerminal, Write},
    sync::Arc,
};

use study_agent::{
    agent::{
        callback::{
            after_tool_callback::search_compressor::SearchCompressor,
            before_llm_callback::{
                context_optimizer::ContextOptimizer, memory_injection::MemoryInjectionCallback,
            },
        },
        context::ExecutionContext,
        event::{ContentItem, ToolResultStatus},
        model::{Agent, AgentStatus},
        runtime::push_user_message,
        tool_confirmation::ToolConfirmation,
    },
    constant::{GPT_6_LUNA_PRO_MODEL, TEXT_EMBEDDING_3_SMALL_MODEL},
    memory::memory_manager::TaskMemoryManager,
    tool::{
        GithubStarsTool, ToolBox, build_basic_toolbox, build_toolbox, code::register_code_tools,
        file::list::ListFileTool, file::read::ReadFileTool, memory_tool::RecallMemoryTool,
    },
};
use termimad::MadSkin;
use uuid::Uuid;

const MEMORY_DB: &str = ".study-agent/memory.db";
const HISTORY_LIMIT: usize = 20;

const HELP: &str = "\
# 交互命令

- `/help` 显示帮助
- `/new` 开始一个新会话
- `/status` 查看模型、会话和功能状态
- `/agents` 查看 agent 树和转移目标
- `/tools` 查看每个 agent 的工具
- `/history [n]` 查看最近 n 条事件，默认 20
- `/remember` 把当前会话提取为长期记忆
- `/clear` 清屏并重画横幅
- `/quit` 退出

直接输入问题即可。输入以 `\\` 结尾时，可以继续下一行。
也可以一次性执行：`cargo run -- 你的问题`。

## 能力

- `transfer_to_agent`：把控制权交给 researcher / math / coder
- 工具审批：危险工具执行前会暂停并询问
- 长期记忆：`/remember` 保存，之后自动检索
- Markdown：回答按终端 Markdown 渲染
";

const ROOT_INSTRUCTIONS: &str = "\
你是运行在终端里的通用 AI 助手。请用中文回答，回答使用 Markdown。

你的团队:
- researcher: 联网搜索、GitHub 数据、最新资料
- math: 数学计算
- coder: 沙箱内 Python / Bash / 文件执行，以及技能

工作规则:
1. 需要最新网络资料时，调用 transfer_to_agent(\"researcher\")。
2. 需要精确数学计算时，调用 transfer_to_agent(\"math\")。
3. 需要执行代码、使用沙箱或技能时，调用 transfer_to_agent(\"coder\")。
4. 文件、规划、反思、记忆、MCP 工具可以直接使用。
5. 能自己完成的任务不要转移；转移后由专家直接回答用户。
6. 不确定时先查证或调用工具，不要编造事实和链接。";

const RESEARCHER_INSTRUCTIONS: &str = "\
你是资料检索专家。请用中文回答。
1. 使用 web_search 查找最新资料，使用 github_stars 查询精确的 GitHub Star 数。
2. 只使用搜索结果中出现过的链接，不要编造来源。
3. 输出简洁的要点列表，并标注来源 URL。
4. 如果问题还包含数学或代码部分，转交给 math 或 coder。";

const MATH_INSTRUCTIONS: &str = "\
你是数学专家。请用中文回答。
1. 使用 calculator 工具计算，不要心算。
2. 展示简要计算步骤，最终答案要明确。
3. 不回答与数学无关的内容。";

const CODER_INSTRUCTIONS: &str = "\
你是代码与沙箱专家。请用中文回答。
1. 使用 execute_python 或 bash_tool 在沙箱中运行代码。
2. 需要读取文件时使用 read_file / list_file；需要上传文件时使用 upload_file。
3. 代码出错时先分析错误，再修正并重新运行。
4. 返回可运行的代码和真实执行结果，不要编造输出。";

struct Features {
    mcp: bool,
    code: bool,
    memory: bool,
    skills: bool,
}

struct Callbacks {
    optimizer: Arc<ContextOptimizer>,
    memory_injection: Option<Arc<MemoryInjectionCallback>>,
    search_compressor: Arc<SearchCompressor>,
}

impl Callbacks {
    fn new(memory: Option<Arc<TaskMemoryManager>>) -> Self {
        Self {
            optimizer: Arc::new(ContextOptimizer::new(GPT_6_LUNA_PRO_MODEL)),
            memory_injection: memory.map(|manager| Arc::new(MemoryInjectionCallback::new(manager))),
            search_compressor: Arc::new(SearchCompressor),
        }
    }

    fn decorate(&self, agent: Agent) -> Agent {
        let mut agent = agent.with_before_llm_callbacks(self.optimizer.clone());
        if let Some(callback) = &self.memory_injection {
            agent = agent.with_before_llm_callbacks(callback.clone());
        }
        agent.with_after_tool_callbacks(self.search_compressor.clone())
    }
}

struct App {
    model: String,
    session_id: String,
    root: Arc<Agent>,
    agents: HashMap<String, Arc<Agent>>,
    context: ExecutionContext,
    memory: Option<Arc<TaskMemoryManager>>,
    features: Features,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("study_agent=info,rmcp=warn"));
    tracing_subscriber::fmt()
        .with_target(false)
        .without_time()
        .with_writer(io::stderr)
        .with_env_filter(filter)
        .init();

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if matches!(args.first().map(String::as_str), Some("-h" | "--help")) {
        render_markdown(HELP);
        return Ok(());
    }

    let mut app = build_app().await?;
    print_banner(&app);

    let one_shot = args.join(" ");
    if !one_shot.trim().is_empty() {
        run_turn(&mut app, one_shot).await?;
        return Ok(());
    }

    loop {
        let Some(input) = read_input()? else {
            break;
        };
        let input = input.trim().to_owned();
        if input.is_empty() {
            continue;
        }

        if input.starts_with('/') {
            if !handle_command(&mut app, &input).await? {
                break;
            }
            continue;
        }

        if let Err(err) = run_turn(&mut app, input).await {
            eprintln!("{} {err:#}", styled("error:", "1;31"));
        }
    }

    println!("\nbye.");
    Ok(())
}

async fn build_app() -> anyhow::Result<App> {
    let model = GPT_6_LUNA_PRO_MODEL.to_owned();

    let (mut available_tools, mcp) = match build_toolbox().await {
        Ok(toolbox) => (toolbox, true),
        Err(err) => {
            eprintln!("warning: MCP unavailable ({err}); using local tools only");
            (build_basic_toolbox(), false)
        }
    };
    // 这个工具是镜像里故意失败的演示工具，交互模式不需要。
    available_tools.remove("get_wikipedia_page");

    let memory = match init_memory(&model) {
        Ok(manager) => Some(Arc::new(manager)),
        Err(err) => {
            eprintln!("warning: long-term memory disabled: {err}");
            None
        }
    };

    let mut math_tools = ToolBox::new();
    if let Some(calculator) = available_tools.remove("calculator") {
        math_tools.insert("calculator".to_owned(), calculator);
    }

    let mut researcher_tools = ToolBox::new();
    if let Some(web_search) = available_tools.remove("web_search") {
        researcher_tools.insert("web_search".to_owned(), web_search);
    }
    researcher_tools.insert("github_stars".to_owned(), Box::new(GithubStarsTool));

    let mut coder_tools = ToolBox::new();
    let sandbox_prompt = match register_code_tools(&mut coder_tools).await {
        Ok(prompt) => prompt,
        Err(err) => {
            eprintln!("warning: code execution disabled: {err}");
            String::new()
        }
    };
    coder_tools.insert("read_file".to_owned(), Box::new(ReadFileTool));
    coder_tools.insert("list_file".to_owned(), Box::new(ListFileTool));

    let code = coder_tools.contains_key("execute_python") || coder_tools.contains_key("bash_tool");
    let skills = sandbox_prompt.contains("Available Skills");

    for toolbox in [
        &mut available_tools,
        &mut researcher_tools,
        &mut math_tools,
        &mut coder_tools,
    ] {
        add_memory_tool(toolbox, &memory);
    }

    let callbacks = Callbacks::new(memory.clone());

    let researcher = Arc::new(callbacks.decorate(make_agent(
        "researcher",
        "联网检索与 GitHub 数据专家",
        RESEARCHER_INSTRUCTIONS.to_owned(),
        researcher_tools,
        12,
    )));
    let math = Arc::new(callbacks.decorate(make_agent(
        "math",
        "数学计算专家",
        MATH_INSTRUCTIONS.to_owned(),
        math_tools,
        6,
    )));

    let mut coder_instructions = CODER_INSTRUCTIONS.to_owned();
    if !sandbox_prompt.is_empty() {
        coder_instructions.push_str("\n\n");
        coder_instructions.push_str(&sandbox_prompt);
    }
    let coder = Arc::new(callbacks.decorate(make_agent(
        "coder",
        "代码执行与沙箱专家",
        coder_instructions,
        coder_tools,
        12,
    )));

    let root = Arc::new(callbacks.decorate(make_agent(
        "assistant",
        "通用助手，负责理解任务并分派专家",
        ROOT_INSTRUCTIONS.to_owned(),
        available_tools,
        24,
    )));
    root.set_sub_agents(vec![researcher.clone(), math.clone(), coder.clone()])?;

    let agents = [
        ("assistant", root.clone()),
        ("researcher", researcher),
        ("math", math),
        ("coder", coder),
    ]
    .into_iter()
    .map(|(name, agent)| (name.to_owned(), agent))
    .collect();

    let session_id = format!("chat-{}", Uuid::new_v4());
    let session = root
        .session_manager
        .get_or_create(&session_id, None)
        .await?;
    let memory_enabled = memory.is_some();

    Ok(App {
        model,
        session_id,
        root,
        agents,
        context: ExecutionContext::new(session),
        memory,
        features: Features {
            mcp,
            code,
            memory: memory_enabled,
            skills,
        },
    })
}

fn init_memory(chat_model: &str) -> anyhow::Result<TaskMemoryManager> {
    std::fs::create_dir_all(".study-agent")?;
    TaskMemoryManager::new(MEMORY_DB, chat_model, TEXT_EMBEDDING_3_SMALL_MODEL)
}

fn add_memory_tool(toolbox: &mut ToolBox, memory: &Option<Arc<TaskMemoryManager>>) {
    if let Some(manager) = memory {
        toolbox.insert(
            "recall_memory".to_owned(),
            Box::new(RecallMemoryTool::new(manager.clone())),
        );
    }
}

fn make_agent(
    name: &str,
    description: &str,
    instructions: String,
    toolbox: ToolBox,
    max_steps: usize,
) -> Agent {
    Agent::new(
        GPT_6_LUNA_PRO_MODEL.to_owned(),
        vec![instructions],
        Arc::new(toolbox),
    )
    .with_name(name.to_owned())
    .with_description(description.to_owned())
    .with_max_steps(max_steps)
}

async fn run_turn(app: &mut App, input: String) -> anyhow::Result<()> {
    push_user_message(&mut app.context, &input);
    // 每个用户回合都是新的一次执行：步数、转移次数和转移路径重新计数。
    app.context.current_step = 0;
    app.context.transfer_count = 0;
    app.context.transfer_history.clear();

    let mut result = app.root.run_in_context(app.context.clone()).await?;
    let mut approval_rounds = 0;

    loop {
        match result.status {
            AgentStatus::Complete => {
                let speaker = result
                    .context
                    .transfer_history
                    .last()
                    .and_then(|hop| hop.rsplit_once(" -> ").map(|(_, to)| to))
                    .unwrap_or(&app.root.name)
                    .to_owned();

                if let Some(output) = result.output {
                    let text = match output {
                        serde_json::Value::String(text) => text,
                        other => format!(
                            "```json\n{}\n```",
                            serde_json::to_string_pretty(&other).unwrap_or_default()
                        ),
                    };
                    println!();
                    println!("{}", styled(&format!("{speaker}>"), "1;32"));
                    render_markdown(&text);
                }

                app.context = result.context;
                app.root
                    .session_manager
                    .save(app.context.session.clone())
                    .await?;
                return Ok(());
            }
            AgentStatus::PendingConfirmation => {
                approval_rounds += 1;
                if approval_rounds > 5 {
                    anyhow::bail!("too many tool approval rounds");
                }

                let active = active_agent(&result.context, &app.root, &app.agents);
                let confirmations = prompt_confirmations(&result.context)?;
                let mut context = result.context;
                context.session.state.insert(
                    "tool_confirmations".to_owned(),
                    serde_json::to_value(&confirmations)?,
                );

                result = active.run_in_context(context).await?;
            }
            AgentStatus::Error => anyhow::bail!("agent stopped with an error"),
        }
    }
}

fn active_agent(
    context: &ExecutionContext,
    root: &Arc<Agent>,
    agents: &HashMap<String, Arc<Agent>>,
) -> Arc<Agent> {
    context
        .transfer_history
        .last()
        .and_then(|hop| hop.rsplit_once(" -> ").map(|(_, to)| to))
        .and_then(|name| agents.get(name))
        .cloned()
        .unwrap_or_else(|| root.clone())
}

fn prompt_confirmations(context: &ExecutionContext) -> anyhow::Result<Vec<ToolConfirmation>> {
    let pending = &context.session.pending_tool_calls;
    if pending.is_empty() {
        return Ok(Vec::new());
    }

    println!("\n{}", styled("tool approval required", "1;33"));
    let mut approve_all = false;
    let mut confirmations = Vec::with_capacity(pending.len());

    for call in pending {
        let approved = if approve_all {
            true
        } else {
            println!("{}", call.confirmation_message);
            print!("[y] approve  [n] deny  [a] approve all: ");
            io::stdout().flush()?;

            let mut reply = String::new();
            if io::stdin().read_line(&mut reply)? == 0 {
                false
            } else {
                match reply.trim().to_lowercase().as_str() {
                    "y" | "yes" => true,
                    "a" | "all" => {
                        approve_all = true;
                        true
                    }
                    _ => false,
                }
            }
        };

        confirmations.push(ToolConfirmation {
            tool_call_id: call.tool_call.id.clone(),
            reason: Some(if approved {
                "approved in terminal".to_owned()
            } else {
                "denied in terminal".to_owned()
            }),
            approved,
            modified_arguments: None,
        });
    }

    Ok(confirmations)
}

async fn handle_command(app: &mut App, input: &str) -> anyhow::Result<bool> {
    let mut parts = input.split_whitespace();
    let command = parts.next().unwrap_or_default();

    match command {
        "/quit" | "/exit" | "/q" => return Ok(false),
        "/help" => render_markdown(HELP),
        "/new" => {
            let session_id = format!("chat-{}", Uuid::new_v4());
            let session = app
                .root
                .session_manager
                .get_or_create(&session_id, None)
                .await?;
            app.context = ExecutionContext::new(session);
            app.session_id = session_id;
            println!("new session: {}", app.session_id);
        }
        "/status" => println!("{}", status_text(app)),
        "/agents" => render_markdown(&agents_markdown(&app.root)),
        "/tools" => render_markdown(&tools_markdown(app)),
        "/history" => {
            let limit = parts
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(HISTORY_LIMIT);
            print_history(&app.context, limit);
        }
        "/remember" | "/save" => remember(app).await?,
        "/clear" => {
            print!("\x1b[2J\x1b[H");
            io::stdout().flush()?;
            print_banner(app);
        }
        other => println!("unknown command: {other}. Type /help for help."),
    }

    Ok(true)
}

async fn remember(app: &mut App) -> anyhow::Result<()> {
    let Some(memory) = &app.memory else {
        println!("long-term memory is disabled");
        return Ok(());
    };

    println!("extracting long-term memory...");
    match memory.save(&app.context.session.events).await {
        Ok(Some(id)) => println!("saved memory: {id}"),
        Ok(None) => println!("nothing saved (duplicate, empty, or extraction failed)"),
        Err(err) => println!("memory save failed: {err:#}"),
    }
    Ok(())
}

fn read_input() -> anyhow::Result<Option<String>> {
    let mut buffer = String::new();
    let terminal = io::stdout().is_terminal();

    loop {
        if terminal {
            if buffer.is_empty() {
                print!("\x1b[1;36myou>\x1b[0m ");
            } else {
                print!("\x1b[1;36m...>\x1b[0m ");
            }
            io::stdout().flush()?;
        }

        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            if buffer.is_empty() {
                return Ok(None);
            }
            break;
        }

        let line = line.trim_end_matches(&['\r', '\n'][..]);
        if let Some(prefix) = line.strip_suffix('\\') {
            buffer.push_str(prefix);
            buffer.push('\n');
            continue;
        }

        buffer.push_str(line);
        break;
    }

    Ok(Some(buffer))
}

fn styled(text: &str, code: &str) -> String {
    if io::stdout().is_terminal() {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_owned()
    }
}

fn render_markdown(text: &str) {
    if io::stdout().is_terminal() {
        let skin = MadSkin::default();
        skin.print_text(text.trim_end());
        println!();
    } else {
        println!("{}", text.trim_end());
    }
}

fn print_banner(app: &App) {
    let memory = if app.features.memory { "on" } else { "off" };
    let code = if app.features.code { "on" } else { "off" };
    let mcp = if app.features.mcp { "on" } else { "off" };
    let skills = if app.features.skills { "on" } else { "off" };

    println!();
    println!("{}  terminal assistant", styled("study-agent", "1;36"));
    println!("model   : {}", app.model);
    println!("session : {}", app.session_id);
    println!("features: mcp={mcp}  code={code}  memory={memory}  skills={skills}  transfer=on");
    let mut agent_names = app.agents.keys().cloned().collect::<Vec<_>>();
    agent_names.sort();
    println!("agents  : {}", agent_names.join(", "));
    println!("commands: /help  /new  /status  /agents  /tools  /history  /remember  /quit");
    println!();
}

fn status_text(app: &App) -> String {
    let transfer = if app.context.transfer_history.is_empty() {
        "(none)".to_owned()
    } else {
        app.context.transfer_history.join(", ")
    };
    format!(
        "model: {}\nsession: {}\nevents: {}\nlast step: {}\ntransfers: {}\nfeatures: mcp={} code={} memory={} skills={}",
        app.model,
        app.session_id,
        app.context.session.events.len(),
        app.context.current_step,
        transfer,
        app.features.mcp,
        app.features.code,
        app.features.memory,
        app.features.skills,
    )
}

fn agents_markdown(root: &Agent) -> String {
    fn walk(agent: &Agent, depth: usize, out: &mut String) {
        let indent = "  ".repeat(depth);
        out.push_str(&format!(
            "{indent}- **{}** — {}\n",
            agent.name, agent.description
        ));

        let targets = agent
            .transfer_targets()
            .iter()
            .map(|target| target.name.clone())
            .collect::<Vec<_>>()
            .join(", ");
        if !targets.is_empty() {
            out.push_str(&format!("{indent}  transfer: `{targets}`\n"));
        }

        for child in agent.sub_agents() {
            walk(child, depth + 1, out);
        }
    }

    let mut out = String::from("# Agents\n\n");
    walk(root, 0, &mut out);
    out
}

fn tools_markdown(app: &App) -> String {
    let mut names = app.agents.keys().cloned().collect::<Vec<_>>();
    names.sort();

    let mut out = String::from("# Tools\n\n");
    for name in names {
        let Some(agent) = app.agents.get(&name) else {
            continue;
        };
        out.push_str(&format!("## {name}\n\n"));

        let mut tools = agent.toolbox.keys().cloned().collect::<Vec<_>>();
        tools.sort();
        for tool_name in tools {
            let description = agent
                .toolbox
                .get(&tool_name)
                .map(|tool| tool.description())
                .unwrap_or_default();
            out.push_str(&format!(
                "- `{tool_name}` — {}\n",
                description.replace('\n', " ")
            ));
        }
        out.push('\n');
    }
    out
}

fn print_history(context: &ExecutionContext, limit: usize) {
    let events = &context.session.events;
    let start = events.len().saturating_sub(limit);
    if start == events.len() {
        println!("(no history)");
        return;
    }

    for event in &events[start..] {
        println!("\n[{}]", event.author);
        for item in &event.content {
            match item {
                ContentItem::Message(message) => {
                    println!("{}: {}", message.role, preview(&message.content, 500));
                }
                ContentItem::ToolCall(call) => {
                    println!("tool_call {} {}", call.name, call.arguments);
                }
                ContentItem::ToolResult(result) => {
                    let status = match result.status {
                        ToolResultStatus::Success => "ok",
                        ToolResultStatus::Error => "error",
                    };
                    println!(
                        "tool_result {} ({status}): {}",
                        result.name,
                        preview(&result.content, 200)
                    );
                }
            }
        }
    }
}

fn preview(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    let mut out = text.chars().take(max_chars).collect::<String>();
    if count > max_chars {
        out.push('…');
    }
    out.replace('\n', " ")
}
