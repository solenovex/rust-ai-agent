use std::sync::Arc;

use ai_agent::{agent::Agent, constant::GPT_4O_MINI_MODEL, tools::build_toolbox};
use chrono::Local;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;
use uuid::Uuid;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv()?;

    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let toolbox = Arc::new(build_toolbox().await?);

    let now = Local::now();
    let current_time = now.format("%Y-%m-%d %H:%M:%S").to_string();

    let instructions = format!(
        r#"你是一位专业、可靠、乐于帮助用户的 AI 助手。

当前本地时间：{}

请始终将"今天"、"昨天"、"明天"、"本周"、"本月"、"上个月"等相对时间，
解释为相对于上面的当前时间。

工具使用原则：
1. 如果问题可以直接回答，则直接回答，不要调用工具。
2. 需要最新信息（新闻、天气、汇率、股票、网络搜索等）时，使用 Web Search 工具。
3. 需要精确计算时，使用 Calculator 工具。
4. 需要查询、统计、新增、修改、删除费用记录时，使用 Expense MCP 提供的工具。
5. 不要猜测工具可以提供的数据，工具能给出答案就调用工具，不要回答"我不知道"。
6. 工具返回结果以后，直接根据结果生成自然、简洁、准确的回答，不要把工具调用过程告诉用户。"#,
        current_time
    );

    let agent = Agent::new(GPT_4O_MINI_MODEL, Some(&instructions), toolbox).with_max_steps(8);

    println!("\n=== Agent::run 测试 ===");
    let result = agent
        .run(
            r"我想买一台 Mac Mini M4。

请帮我做一个购买分析：
1. 使用搜索工具查询目前 Mac Mini M4 的价格。
2. 查询我过去三个月的 Software 分类支出。
3. 计算 Mac Mini M4 价格占过去三个月 Software 支出的多少倍，
   以及每个月节省 500 元需要多少个月才能攒够。
4. 根据我的消费情况，给我一个是否应该购买的建议。

所有价格和消费数据必须来自工具，不要自己猜测数据。",
            &Uuid::new_v4().to_string(),
        )
        .await?;

    println!("回答: {}", result.output);
    println!(
        "\n本次执行一共走了 {} 步，记录了 {} 条 Event（execution_id = {}）",
        result.context.current_step,
        result.context.events().len(),
        result.context.execution_id
    );

    println!(
        "token 用量：prompt={} completion={} total={}",
        result.context.usage.prompt_tokens,
        result.context.usage.completion_tokens,
        result.context.usage.total_tokens
    );

    println!("{:#?}", result.context);

    Ok(())
}
