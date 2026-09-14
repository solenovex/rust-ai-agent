use std::{
    io::{self, Write},
    sync::Arc,
};

use ai_agent::{
    agent::{
        Agent, AgentStatus, ContentItem, ExecutionContext, PendingToolCall, TokenUsage,
        ToolConfirmation,
    },
    callback::{context_optimizer::ContextOptimizer, search_compressor::SearchCompressorCallback},
    constant::{GPT_4O_MINI_MODEL, VISION_MODEL},
    tools::{build_file_explorer_toolbox, build_toolbox},
};
use chrono::Local;
use uuid::Uuid;

const HELP: &str = r#"命令：
  /help       显示帮助
  /new        开始一个新的会话
  /session    显示当前会话 ID
  /quit       退出

直接输入问题即可。助手可以搜索网页、计算、管理费用、探索文件，
并在执行删除文件前请求确认。"#;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_target(false)
        .with_max_level(tracing::Level::WARN)
        .init();

    println!("Rust AI Agent");
    println!("输入 /help 查看命令，输入 /quit 退出。\n");
    println!("正在连接工具...");

    // Combine the tutorial's general and file-exploration toolboxes into the
    // complete toolbox used by the terminal application.
    let mut toolbox = build_toolbox().await?;
    toolbox.extend(build_file_explorer_toolbox(VISION_MODEL));
    let toolbox = Arc::new(toolbox);

    let now = Local::now().format("%Y-%m-%d %H:%M:%S");
    let instructions = format!(
        r#"你是一位可靠、简洁、会主动使用工具的终端 AI 助手。

当前本地时间：{now}
请将“今天、昨天、明天、本周、本月”等相对时间解释为相对于这个时间。

工具原则：
1. 可以直接回答的问题直接回答；需要最新信息时使用 web_search。
2. 需要精确计算时使用 calculator，不要心算猜测。
3. 涉及费用数据时使用 Expense MCP 工具，不要编造记录。
4. 探索压缩包或文件时，先 list_files/read_file/read_image 获取事实再下结论。
5. 删除文件是不可逆操作，只有用户明确要求且确认后才执行。
6. 工具返回后直接给出自然、准确的结论，不要泄露内部推理。"#
    );

    let agent = Agent::new(GPT_4O_MINI_MODEL, Some(instructions), toolbox)
        .with_max_steps(12)
        .with_after_tool_callback(Arc::new(SearchCompressorCallback))
        .with_before_llm_callback(Arc::new(ContextOptimizer {
            token_threshold: 20_000,
            compaction_keep_recent: 6,
            keep_recent: 6,
            ..ContextOptimizer::new(GPT_4O_MINI_MODEL)
        }));

    let mut session_id = new_session_id();
    let mut session_usage = TokenUsage::default();
    loop {
        print!("\n你 [{}]> ", short_session_id(&session_id));
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            println!("\n再见！");
            break;
        }
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        match input {
            "/help" => {
                println!("\n{HELP}");
                continue;
            }
            "/quit" | "/exit" => {
                println!("再见！");
                break;
            }
            "/new" => {
                session_id = new_session_id();
                session_usage = TokenUsage::default();
                println!("已开始新会话：{}", short_session_id(&session_id));
                continue;
            }
            "/session" => {
                println!("当前会话：{session_id}");
                continue;
            }
            command if command.starts_with('/') => {
                println!("未知命令：{command}（输入 /help 查看帮助）");
                continue;
            }
            prompt => {
                println!("\n助手正在处理...");
                if let Err(error) = run_turn(&agent, prompt, &session_id, &mut session_usage).await
                {
                    eprintln!("\n执行失败：{error:#}");
                    eprintln!("你可以继续输入问题，或使用 /new 开始新会话。\n");
                }
            }
        }
    }

    Ok(())
}

fn new_session_id() -> String {
    Uuid::new_v4().to_string()
}

fn short_session_id(session_id: &str) -> &str {
    session_id.get(..8).unwrap_or(session_id)
}

async fn run_turn(
    agent: &Agent,
    prompt: &str,
    session_id: &str,
    session_usage: &mut TokenUsage,
) -> anyhow::Result<()> {
    let result = agent.run(Some(prompt), session_id, None).await?;
    *session_usage = add_usage(*session_usage, result.context.usage);
    print_tool_trace(&result.context);

    if result.status == AgentStatus::PendingConfirmation {
        let confirmations = ask_for_confirmations(&result.pending_tool_calls)?;
        let resumed = agent.run(None, session_id, Some(confirmations)).await?;
        *session_usage = add_usage(*session_usage, resumed.context.usage);
        print_tool_trace(&resumed.context);
        print_result(&resumed, session_usage);
    } else {
        print_result(&result, session_usage);
    }

    Ok(())
}

fn ask_for_confirmations(
    pending_tool_calls: &[PendingToolCall],
) -> anyhow::Result<Vec<ToolConfirmation>> {
    let mut confirmations = Vec::with_capacity(pending_tool_calls.len());

    for pending in pending_tool_calls {
        println!("\n⚠️  需要确认");
        println!("{}", pending.confirmation_message);
        print!("确认执行？(y/n): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let approved = input.trim().eq_ignore_ascii_case("y");
        confirmations.push(ToolConfirmation {
            tool_call_id: pending.tool_call.tool_call_id.clone(),
            reason: (!approved).then(|| "用户拒绝了这次操作".to_string()),
            approved,
            modified_arguments: None,
        });
    }

    Ok(confirmations)
}

fn print_result(result: &ai_agent::agent::AgentResult, session_usage: &TokenUsage) {
    println!(
        "\n助手：{}",
        result
            .output
            .clone()
            .unwrap_or_else(|| "这次操作没有生成最终回答。".to_string())
    );
    println!(
        "（本次 {} 步，本次 {} tokens；会话累计 prompt={} completion={} total={} tokens）",
        result.context.current_step,
        result.context.usage.total_tokens,
        session_usage.prompt_tokens,
        session_usage.completion_tokens,
        session_usage.total_tokens,
    );
}

fn add_usage(mut total: TokenUsage, current: TokenUsage) -> TokenUsage {
    total.add(
        current.prompt_tokens,
        current.completion_tokens,
        current.total_tokens,
    );
    total
}

fn print_tool_trace(context: &ExecutionContext) {
    for event in context.events() {
        if event.execution_id != context.execution_id {
            continue;
        }

        for item in &event.content {
            match item {
                ContentItem::ToolCall(tool_call) => {
                    println!(
                        "\n🔧 调用工具：{}\n   参数：{}",
                        tool_call.name, tool_call.arguments
                    );
                }
                ContentItem::ToolResult {
                    name,
                    status,
                    content,
                    ..
                } => {
                    let char_count = content.chars().count();
                    let preview: String = content.chars().take(500).collect();
                    let suffix = if char_count > 500 { "..." } else { "" };
                    println!("   工具结果：{name} [{status:?}]\n   {preview}{suffix}");
                }
                ContentItem::Message { .. } => {}
            }
        }
    }
}
