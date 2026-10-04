use std::sync::{Arc, Mutex};

use crate::{
    agent::{callback::before_llm_callback::BeforeLlmCallback, event::ContentItem},
    llm::llm_request::LlmRequest,
    memory::memory_manager::TaskMemoryManager,
    session::model::Session,
    tool::memory_tool::RecallMemoryTool,
};

/// 每次注入检索的记录条数
const DEFAULT_TOP_K: usize = 3;

/// 缓存的检索结果:查询词 + 格式化后的记忆文本(没搜到则为 None)
type CachedRecall = (String, Option<String>);

/// 在每次调用 LLM 之前,自动检索相关的历史解题记录并注入 instructions。
///
/// 与 RecallMemoryTool(工具式检索)的区别:这里不进 ToolBox,LLM
/// 看不到、也无法主动调用它;它对每一轮 LLM 请求都会运行,
/// 把最近一条用户消息当作查询词去检索。两者共享同一份
/// Arc<TaskMemoryManager>,也复用同一套格式化逻辑
/// (RecallMemoryTool::format_memories)。
///
/// 注入的内容只存在于当次请求里(每一步的 LlmRequest 都是重新构造的),
/// 所以每一步都必须重新注入;但同一个查询词只检索一次,
/// 之后的步骤直接复用缓存,避免重复的 embedding 请求。
pub struct MemoryInjectionCallback {
    memory_manager: Arc<TaskMemoryManager>,
    top_k: usize,
    cache: Mutex<Option<CachedRecall>>,
}

impl MemoryInjectionCallback {
    pub fn new(memory_manager: Arc<TaskMemoryManager>) -> Self {
        Self {
            memory_manager,
            top_k: DEFAULT_TOP_K,
            cache: Mutex::new(None),
        }
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    fn latest_user_message(request: &LlmRequest) -> Option<String> {
        request.contents.iter().rev().find_map(|item| match item {
            ContentItem::Message(msg) if msg.role == "user" => Some(msg.content.clone()),
            _ => None,
        })
    }

    /// 命中缓存时返回 Some(结果),结果本身可能是 None(上次没搜到)
    fn cached(&self, query: &str) -> Option<Option<String>> {
        let guard = self.cache.lock().ok()?;
        match guard.as_ref() {
            Some((q, text)) if q == query => Some(text.clone()),
            _ => None,
        }
    }

    fn store(&self, query: &str, text: Option<String>) {
        if let Ok(mut guard) = self.cache.lock() {
            *guard = Some((query.to_string(), text));
        }
    }

    /// 取得该查询词对应的记忆文本:先查缓存,未命中再检索。
    /// 注意:MutexGuard 不能跨 await 持有,所以查缓存和写缓存是分开的函数。
    async fn recall(&self, query: &str) -> Option<String> {
        if let Some(hit) = self.cached(query) {
            return hit;
        }

        match self.memory_manager.search(query, self.top_k).await {
            Ok(memories) if !memories.is_empty() => {
                let text = RecallMemoryTool::format_memories(&memories);
                self.store(query, Some(text.clone()));
                Some(text)
            }
            Ok(_) => {
                self.store(query, None);
                None
            }
            Err(err) => {
                // 失败不缓存,下一步还会重试
                tracing::warn!("memory injection search failed: {err}");
                None
            }
        }
    }
}

#[async_trait::async_trait]
impl BeforeLlmCallback for MemoryInjectionCallback {
    async fn before_llm_call(&self, _session: &mut Session, request: &mut LlmRequest) {
        let Some(query) = Self::latest_user_message(request) else {
            return;
        };

        if let Some(formatted) = self.recall(&query).await {
            tracing::info!("injected past experiences:\n{formatted}");
            request.append_instruction(format!(
                "以下是过去解决类似问题的记录:\n\n<PAST_EXPERIENCES>\n{formatted}\n</PAST_EXPERIENCES>\n\n请参考成功的方法,避免重复失败的尝试。"
            ));
        }
    }
}
