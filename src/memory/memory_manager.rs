use crate::{
    agent::event::{AgentEvent, ContentItem},
    llm::ask::ask,
    memory::{
        prompt::{DUPLICATE_CHECK_PROMPT, TASK_MEMORY_EXTRACTION_PROMPT},
        store::TaskMemoryStore,
        task_memory::{DuplicateCheckResult, TaskMemory},
    },
    search::embedding::embed_text,
};

/// 相似度低于此值时,直接认定不重复,跳过 LLM 查重(需要用真实数据调)
const DUPLICATE_CANDIDATE_THRESHOLD: f32 = 0.8;
/// search() 只返回相似度不低于此值的记忆,避免无关记忆污染上下文
const SEARCH_MIN_SIMILARITY: f32 = 0.5;
/// 查重时最多取多少条近邻
const DUPLICATE_CHECK_TOP_K: usize = 3;
/// 工具结果在执行历史里最多保留的字符数
const TOOL_RESULT_PREVIEW_CHARS: usize = 500;

pub struct TaskMemoryManager {
    store: TaskMemoryStore,
    chat_model: String,
    embedding_model: String,
}

impl TaskMemoryManager {
    pub fn new(
        db_path: &str,
        chat_model: impl Into<String>,
        embedding_model: impl Into<String>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            store: TaskMemoryStore::open(db_path)?,
            chat_model: chat_model.into(),
            embedding_model: embedding_model.into(),
        })
    }

    /// 提取并保存记忆。返回 Some(id) 表示已保存,None 表示被跳过
    /// (提取失败、嵌入失败或重复)。记忆是辅助功能,这些情况都不报错。
    pub async fn save(&self, events: &[AgentEvent]) -> anyhow::Result<Option<String>> {
        let execution_history = Self::format_execution_history(events);

        let memory = match self.extract_memory(&execution_history).await {
            Ok(memory) => memory,
            Err(err) => {
                tracing::warn!("memory extraction failed: {err}");
                return Ok(None);
            }
        };

        // embedding 只算一次,查重和存储共用
        let embedding = match embed_text(&memory.to_embedding_text(), &self.embedding_model).await {
            Ok(embedding) => embedding,
            Err(err) => {
                tracing::warn!("memory embedding failed: {err}");
                return Ok(None);
            }
        };

        // 只有足够相似的才值得让 LLM 判断是否重复
        let candidates: Vec<TaskMemory> = self
            .store
            .query(&embedding, DUPLICATE_CHECK_TOP_K)?
            .into_iter()
            .filter(|(score, _)| *score >= DUPLICATE_CANDIDATE_THRESHOLD)
            .map(|(_, mem)| mem)
            .collect();

        if !candidates.is_empty() && self.is_duplicate(&memory, &candidates).await {
            return Ok(None);
        }

        let memory_id = uuid::Uuid::new_v4().to_string();
        self.store.add(&memory_id, &memory, &embedding)?;
        Ok(Some(memory_id))
    }

    pub async fn search(&self, query: &str, top_k: usize) -> anyhow::Result<Vec<TaskMemory>> {
        let embedding = embed_text(query, &self.embedding_model).await?;
        Ok(self
            .store
            .query(&embedding, top_k)?
            .into_iter()
            .filter(|(score, _)| *score >= SEARCH_MIN_SIMILARITY)
            .map(|(_, mem)| mem)
            .collect())
    }

    async fn extract_memory(&self, execution_history: &str) -> anyhow::Result<TaskMemory> {
        let prompt =
            TASK_MEMORY_EXTRACTION_PROMPT.replace("{execution_history}", execution_history);
        ask::<TaskMemory>(&self.chat_model, &prompt).await
    }

    /// 查重失败时默认返回 false(继续存储),所以不返回 Result
    async fn is_duplicate(&self, new_memory: &TaskMemory, existing: &[TaskMemory]) -> bool {
        let existing_text = existing
            .iter()
            .map(Self::render_for_dedup)
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = DUPLICATE_CHECK_PROMPT
            .replace("{existing_memories}", &existing_text)
            .replace("{new_memory}", &Self::render_for_dedup(new_memory));

        match ask::<DuplicateCheckResult>(&self.chat_model, &prompt).await {
            Ok(result) => result.is_duplicate(),
            Err(err) => {
                tracing::warn!("duplicate check failed, proceeding to store: {err}");
                false
            }
        }
    }

    /// 查重 prompt 的判断标准包含"同结果",所以必须带上 final_answer
    fn render_for_dedup(mem: &TaskMemory) -> String {
        format!(
            "task_summary: {}, approach: {}, final_answer: {}, is_correct: {}",
            mem.task_summary, mem.approach, mem.final_answer, mem.is_correct
        )
    }

    fn format_execution_history(events: &[AgentEvent]) -> String {
        let mut lines = Vec::new();
        for event in events {
            for item in &event.content {
                match item {
                    ContentItem::Message(msg) => {
                        lines.push(format!("[{}]: {}", msg.role, msg.content));
                    }
                    ContentItem::ToolCall(tc) => {
                        lines.push(format!("[Tool Call]: {}({})", tc.name, tc.arguments));
                    }
                    ContentItem::ToolResult(tr) => {
                        let preview: String =
                            tr.content.chars().take(TOOL_RESULT_PREVIEW_CHARS).collect();
                        lines.push(format!("[Tool Result]: {} -> {preview}", tr.name));
                    }
                }
            }
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod memory_manager_tests {
    use crate::agent::event::{Message, ToolCall, ToolResult, ToolResultStatus};

    use super::*;

    #[test]
    fn format_execution_history_renders_all_content_kinds() {
        let events = vec![AgentEvent::new(
            "exec-1",
            "user",
            vec![
                ContentItem::Message(Message::new("user".to_string(), "帮我算一下".to_string())),
                ContentItem::ToolCall(ToolCall {
                    tool_call_id: "call-1".to_string(),
                    name: "calculator".to_string(),
                    arguments: serde_json::json!({"expr": "1+1"}),
                }),
                ContentItem::ToolResult(ToolResult::new(
                    "call-1".to_string(),
                    "calculator".to_string(),
                    ToolResultStatus::Success,
                    "2".to_string(),
                )),
            ],
        )];

        let text = TaskMemoryManager::format_execution_history(&events);

        assert!(text.contains("[user]: 帮我算一下"));
        assert!(text.contains("[Tool Call]: calculator("));
        assert!(text.contains("[Tool Result]: calculator -> 2"));
    }

    #[test]
    fn render_for_dedup_includes_final_answer() {
        let mem = TaskMemory {
            task_summary: "圆的面积".to_string(),
            approach: "π r²".to_string(),
            final_answer: "78.5".to_string(),
            is_correct: true,
            error_analysis: None,
        };
        let text = TaskMemoryManager::render_for_dedup(&mem);
        assert!(text.contains("final_answer: 78.5"));
    }
}
