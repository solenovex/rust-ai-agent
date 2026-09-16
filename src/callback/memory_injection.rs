use std::sync::Arc;

use crate::{
    agent::{
        llm_request::{BeforeLlmCallback, LlmRequest},
        ContentItem, ExecutionContext,
    },
    memory::manager::TaskMemoryManager,
    tools::recall_memory::execute::RecallMemoryTool,
};

/// 在每次调用 LLM 之前，自动检索相关的历史解题记录并注入 instructions。
///
/// 与 RecallMemoryTool（工具式检索）的区别：这里不进 ToolBox，LLM
/// 看不到、也无法主动调用它；它对每一轮 LLM 请求都无条件运行一次，
/// 把最近一条用户消息当作查询词去检索。两者共享同一份
/// Arc<TaskMemoryManager>，也复用同一套格式化逻辑
/// （RecallMemoryTool::format_memories）。
pub struct MemoryInjectionCallback {
    memory_manager: Arc<TaskMemoryManager>,
    top_k: usize,
}

impl MemoryInjectionCallback {
    pub fn new(memory_manager: Arc<TaskMemoryManager>) -> Self {
        Self {
            memory_manager,
            top_k: 3,
        }
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    fn latest_user_message(request: &LlmRequest) -> Option<String> {
        request.contents.iter().rev().find_map(|item| match item {
            ContentItem::Message { role, content } if role == "user" => Some(content.clone()),
            _ => None,
        })
    }
}

#[async_trait::async_trait]
impl BeforeLlmCallback for MemoryInjectionCallback {
    async fn call(&self, _context: &mut ExecutionContext, request: &mut LlmRequest) {
        let Some(query) = Self::latest_user_message(request) else {
            return;
        };

        let memories = match self.memory_manager.search(&query, self.top_k).await {
            Ok(memories) => memories,
            Err(err) => {
                tracing::warn!("memory injection search failed: {err}");
                return;
            }
        };

        if memories.is_empty() {
            return;
        }

        let formatted = RecallMemoryTool::format_memories(&memories);
        request.append_instructions(format!(
            "以下是过去解决类似问题的记录：\n\n<PAST_EXPERIENCES>\n{formatted}\n</PAST_EXPERIENCES>\n\n请参考成功的方法，避免重复失败的尝试。"
        ));
    }
}
