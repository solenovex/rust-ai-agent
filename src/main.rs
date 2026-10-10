use std::sync::Arc;

use study_agent::{
    agent::{
        callback::before_llm_callback::{
            context_optimizer::ContextOptimizer, memory_injection::MemoryInjectionCallback,
        },
        event::{AgentEvent, ContentItem, Message},
        model::Agent,
    }, constant::{FINAL_ANSWER_TOOL_NAME, GPT_6_LUNA_PRO_MODEL, TEXT_EMBEDDING_3_SMALL_MODEL}, llm::llm_answer::TextAnswer, memory::memory_manager::TaskMemoryManager, tool::{Tool, build_toolbox, final_answer::FinalAnswerTool, memory_tool::RecallMemoryTool},
};
use uuid::Uuid;

const DB_PATH: &str = "memory_test.db";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    tracing_subscriber::fmt()
        .with_target(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    // 每次从空库开始,结果才可重复
    let _ = std::fs::remove_file(DB_PATH);

    let memory_manager = Arc::new(TaskMemoryManager::new(
        DB_PATH,
        GPT_6_LUNA_PRO_MODEL,
        TEXT_EMBEDDING_3_SMALL_MODEL,
    )?);

    let mut toolbox = build_toolbox().await?;
    toolbox.insert(
        FINAL_ANSWER_TOOL_NAME.to_owned(),
        Box::new(FinalAnswerTool::<TextAnswer>::new()),
    );    
    let recall_memory_tool = RecallMemoryTool::new(memory_manager.clone());
    toolbox.insert(
        recall_memory_tool.name().to_owned(),
        Box::new(recall_memory_tool),
    );

    let agent = Agent::new(
        GPT_6_LUNA_PRO_MODEL.to_owned(),
        vec!["You are a helpful study assistant for children.".to_owned()],
        Arc::new(toolbox),
    )
    .with_max_steps(10)
    // 记忆注入放在 ContextOptimizer 之前
    .with_before_llm_callbacks(Arc::new(MemoryInjectionCallback::new(
        memory_manager.clone(),
    )))
    .with_before_llm_callbacks(Arc::new(
        ContextOptimizer::new(GPT_6_LUNA_PRO_MODEL).with_token_threshold(1000),
    ));

    // ============================================================
    // STEP 1: 会话 A,做一道题
    // ============================================================
    println!("\n========== STEP 1: 会话 A 做题 ==========\n");

    let session_a = Uuid::new_v4().to_string();
    let result = agent
        .run::<TextAnswer>(
            "一个圆的直径是 10 厘米,它的面积是多少?(π 取 3.14)",
            &session_a,
            None,
        )
        .await?;
    println!("Result: {result}");

    // ============================================================
    // STEP 2: 追加对错反馈,保存记忆
    // ============================================================
    println!("\n========== STEP 2: 保存记忆 ==========\n");

    let mut events = result.context.session.events.clone();
    events.push(AgentEvent::new(
        result.context.execution_id.clone(),
        "user",
        vec![ContentItem::Message(Message::new(
            "user".to_string(),
            "[家长反馈] 答对了,正确答案是 78.5 平方厘米,方法是先把直径除以 2 得到半径,再用 π r²。"
                .to_string(),
        ))],
    ));

    let saved = memory_manager.save(&events).await?;
    println!("第一次 save: {saved:?}   (期望 Some(id))");

    let saved_again = memory_manager.save(&events).await?;
    println!("第二次 save: {saved_again:?}   (期望 None,被查重跳过)");

    // ============================================================
    // STEP 3: 直接检索,验证记忆确实存进去了
    // ============================================================
    println!("\n========== STEP 3: 直接检索 ==========\n");

    let found = memory_manager.search("半径是 3 厘米的圆,面积多少?", 3).await?;
    println!("检索到 {} 条", found.len());
    for mem in &found {
        println!("{mem:#?}");
    }

    // ============================================================
    // STEP 4: 新会话,问类似的题,看有没有自动注入
    // ============================================================
    println!("\n========== STEP 4: 新会话问类似的题 ==========\n");

    let session_b = Uuid::new_v4().to_string();
    let result = agent
        .run::<TextAnswer>(
            "一个圆的直径是 20 厘米,它的面积是多少?(π 取 3.14)",
            &session_b,
            None,
        )
        .await?;
    println!("Result: {result}");

    Ok(())
}