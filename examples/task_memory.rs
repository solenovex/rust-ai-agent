use ai_agent::{
    agent::event::{ContentItem, Event, ToolCall, ToolResultStatus},
    constant::{GPT_4O_MINI_MODEL, TEXT_EMBEDDING_3_SMALL_MODEL},
    memory::manager::TaskMemoryManager,
};

/// 验证长期记忆模块：先模拟一次“解题过程”存成记忆，再用一个相近但
/// 不完全相同的问法去检索，确认语义检索（而不是关键词匹配）真的生效。
///
/// 运行：
///   cargo run --example task_memory_demo
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_target(false)
        .with_max_level(tracing::Level::INFO)
        .init();

    // 用一个临时数据库文件，方便重复运行时观察查重是否生效；
    // 想要每次全新开始，运行前手动删掉这个文件即可。
    let db_path = "task_memories.db";
    let manager = TaskMemoryManager::new(
        db_path,
        GPT_4O_MINI_MODEL,
        TEXT_EMBEDDING_3_SMALL_MODEL,
    )?;

    println!("=== 第一步：模拟一次解题过程，调用 save() ===\n");

    let events = vec![Event::new(
        "demo-exec-1",
        "user",
        vec![
            ContentItem::Message {
                role: "user".to_string(),
                content: "一个半径为 5 厘米的圆，面积是多少？".to_string(),
            },
            ContentItem::ToolCall(ToolCall {
                tool_call_id: "call-1".to_string(),
                name: "calculator".to_string(),
                arguments: serde_json::json!({ "expr": "3.14159 * 5 * 5" }),
            }),
            ContentItem::ToolResult {
                tool_call_id: "call-1".to_string(),
                name: "calculator".to_string(),
                status: ToolResultStatus::Success,
                content: "78.53975".to_string(),
            },
            ContentItem::Message {
                role: "assistant".to_string(),
                content: "圆的面积约为 78.54 平方厘米，用了公式 π × r²。".to_string(),
            },
        ],
    )];

    match manager.save(&events).await? {
        Some(memory_id) => println!("已存储新记忆，id = {memory_id}"),
        None => println!("被判定为重复记忆，跳过存储（如果是第二次运行，这是预期行为）"),
    }

    println!("\n=== 第二步：用一个相近但不同的问法去检索 ===\n");

    // 故意不用“圆”“面积”这两个字，改问“花坛周长”，
    // 验证靠的是语义相似度而不是关键词命中。
    let query = "一个圆形花坛，半径是 5 米，铺草皮大概要多少面积？";
    let results = manager.search(query, 3).await?;

    print_memories(&results);

    println!("\n=== 第三步：模拟一次“答错了”的执行过程，验证 error_analysis 这条从未走过的路径 ===\n");

    // 之前跑过的例子 is_correct 恒为 true，从没验证过
    // error_analysis: Option<String> 在 Some(...) 情况下能否正常经过
    // “LLM 抽取 -> 序列化进 SQLite -> 再反序列化读出来”这一整条链路。
    // 这里故意用错公式（周长公式 2πr，而不是面积公式 πr²），并让
    // 用户在对话里指出算错了，逼 LLM 把这次尝试标记为 is_correct: false，
    // 并写出 error_analysis。
    let wrong_events = vec![Event::new(
        "demo-exec-2",
        "user",
        vec![
            ContentItem::Message {
                role: "user".to_string(),
                content: "一个半径为 8 米的圆形水池，面积是多少平方米？".to_string(),
            },
            ContentItem::ToolCall(ToolCall {
                tool_call_id: "call-2".to_string(),
                name: "calculator".to_string(),
                arguments: serde_json::json!({ "expr": "2 * 3.14159 * 8" }),
            }),
            ContentItem::ToolResult {
                tool_call_id: "call-2".to_string(),
                name: "calculator".to_string(),
                status: ToolResultStatus::Success,
                content: "50.26544".to_string(),
            },
            ContentItem::Message {
                role: "assistant".to_string(),
                content: "水池的面积约为 50.27 平方米。".to_string(),
            },
            ContentItem::Message {
                role: "user".to_string(),
                content: "不对，你算的是周长公式 2πr，我要的是面积，应该用 πr² 才对。正确答案约是 201.06 平方米。".to_string(),
            },
        ],
    )];

    match manager.save(&wrong_events).await? {
        Some(memory_id) => println!("已存储新记忆（预期是一次失败记录），id = {memory_id}"),
        None => println!("被判定为重复记忆，跳过存储（如果是第二次运行，这是预期行为）"),
    }

    println!("\n=== 第四步：检索刚才这条失败记录，确认 error_analysis 完整存取 ===\n");

    let wrong_query = "圆形水池的面积怎么算";
    let wrong_results = manager.search(wrong_query, 3).await?;
    print_memories(&wrong_results);

    println!("提示：数据库文件位于 ./{db_path}，删除它可以重新从空白状态开始。");

    Ok(())
}

fn print_memories(results: &[ai_agent::memory::task_memory::TaskMemory]) {
    if results.is_empty() {
        println!("没有检索到相关记忆。");
        return;
    }

    println!("检索到 {} 条相关记忆：\n", results.len());
    for (i, mem) in results.iter().enumerate() {
        println!("[记录 {}]", i + 1);
        println!("  问题: {}", mem.task_summary);
        println!("  方法: {}", mem.approach);
        println!("  答案: {}", mem.final_answer);
        println!("  正确: {}", mem.is_correct);
        match &mem.error_analysis {
            Some(analysis) => println!("  错误分析: {analysis}"),
            None => println!("  错误分析: (无，说明这条是成功记录)"),
        }
        println!();
    }
}
