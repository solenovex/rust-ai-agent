use crate::{
    agent::{ContentItem, Event},
    knowledge_base::embed::embed_text,
    llm::ask::ask_structured,
    memory::{
        prompt::{DUPLICATE_CHECK_PROMPT, TASK_MEMORY_EXTRACTION_PROMPT},
        store::TaskMemoryStore,
        task_memory::{DuplicateCheckResult, TaskMemory},
    },
};

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

    pub async fn save(&self, events: &[Event]) -> anyhow::Result<Option<String>> {
        let execution_history = Self::format_execution_history(events);
        let memory = match self.extract_memory(&execution_history).await {
            Ok(memory) => memory,
            Err(err) => {
                tracing::warn!("memory extraction failed: {err}");
                return Ok(None);
            }
        };

        let embedding = embed_text(&memory.to_embedding_text(), &self.embedding_model).await?;
        let similar = self.store.query(&embedding, 3)?;

        if !similar.is_empty() && self.is_duplicate(&memory, &similar).await? {
            return Ok(None);
        }

        let memory_id = uuid::Uuid::new_v4().to_string();
        self.store.add(&memory_id, &memory, &embedding)?;
        Ok(Some(memory_id))
    }

    pub async fn search(&self, query: &str, top_k: usize) -> anyhow::Result<Vec<TaskMemory>> {
        let embedding = embed_text(query, &self.embedding_model).await?;
        self.store.query(&embedding, top_k)
    }

    async fn extract_memory(&self, execution_hisotry: &str) -> anyhow::Result<TaskMemory> {
        let prompt =
            TASK_MEMORY_EXTRACTION_PROMPT.replace("{execution_history}", execution_hisotry);
        ask_structured::<TaskMemory>(&self.chat_model, &prompt).await
    }

    async fn is_duplicate(
        &self,
        new_memory: &TaskMemory,
        existing: &[TaskMemory],
    ) -> anyhow::Result<bool> {
        let existing_texts: Vec<String> = existing
            .iter()
            .map(|mem| {
                format!(
                    "task_summary: {}, approach: {}, is_correct: {}",
                    mem.task_summary, mem.approach, mem.is_correct
                )
            })
            .collect();

        let new_text = format!(
            "task_summary: {}, approach: {}, is_correct: {}",
            new_memory.task_summary, new_memory.approach, new_memory.is_correct
        );

        let prompt = DUPLICATE_CHECK_PROMPT
            .replace("{existing_memories}", &existing_texts.join("\n"))
            .replace("{new_memory}", &new_text);

        match ask_structured::<DuplicateCheckResult>(&self.chat_model, &prompt).await {
            Ok(result) => Ok(result.is_duplicate()),
            Err(err) => {
                tracing::warn!("duplicate check failed, proceeding to store: {err}");
                Ok(false)
            }
        }
    }

    fn format_execution_history(events: &[Event]) -> String {
        let mut lines = Vec::new();
        for event in events {
            for item in &event.content {
                match item {
                    ContentItem::Message { role, content } => {
                        lines.push(format!("[{role}]: {content}"));
                    }
                    ContentItem::ToolCall(call) => {
                        lines.push(format!("[Tool Call]: {}({})", call.name, call.arguments));
                    }
                    ContentItem::ToolResult { name, content, .. } => {
                        let preview: String = content.chars().take(500).collect();
                        lines.push(format!("[Tool Result]: {name} -> {preview}"));
                    }
                }
            }
        }

        lines.join("\n")
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_execution_history_renders_all_content_kinds() {
        let events = vec![Event::new(
            "exec-1",
            "user",
            vec![
                ContentItem::Message {
                    role: "user".to_string(),
                    content: "帮我算一下".to_string(),
                },
                ContentItem::ToolCall(crate::agent::event::ToolCall {
                    tool_call_id: "call-1".to_string(),
                    name: "calculator".to_string(),
                    arguments: serde_json::json!({"expr": "1+1"}),
                }),
                ContentItem::ToolResult {
                    tool_call_id: "call-1".to_string(),
                    name: "calculator".to_string(),
                    status: crate::agent::event::ToolResultStatus::Success,
                    content: "2".to_string(),
                },
            ],
        )];

        let text = TaskMemoryManager::format_execution_history(&events);

        assert!(text.contains("[user]: 帮我算一下"));
        assert!(text.contains("[Tool Call]: calculator("));
        assert!(text.contains("[Tool Result]: calculator -> 2"));
    }
}
