use serde_json::{json, Value};

use crate::{agent::ExecutionContext, tools::{recall_memory::execute::RecallMemoryTool, tool::Tool}};

#[async_trait::async_trait]
impl Tool for RecallMemoryTool {
    fn name(&self) -> &str {
        "recall_memory"
    }

    fn description(&self) -> &str {
        "搜索过往的解题记录，用于检查是否遇到过类似的问题"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "要搜索的问题描述" }
            },
            "required": ["query"]
        })
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(args_json)?;
        let query = args["query"].as_str().unwrap_or_default();

        let memories = self.memory_manager.search(query, 3).await?;
        if memories.is_empty() {
            return Ok(String::new());
        }

        Ok(Self::format_memories(&memories))
    }
}
