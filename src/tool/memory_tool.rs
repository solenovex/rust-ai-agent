use std::sync::Arc;

use serde::Deserialize;
use serde_json::json;

use crate::{
    agent::context::ExecutionContext,
    memory::{memory_manager::TaskMemoryManager, task_memory::TaskMemory},
    tool::Tool,
};

/// 每次召回返回的记录条数
const RECALL_TOP_K: usize = 3;

#[derive(Deserialize)]
struct RecallArgs {
    query: String,
}

pub struct RecallMemoryTool {
    memory_manager: Arc<TaskMemoryManager>,
}

impl RecallMemoryTool {
    pub fn new(memory_manager: Arc<TaskMemoryManager>) -> Self {
        Self { memory_manager }
    }

    pub fn format_memories(memories: &[TaskMemory]) -> String {
        memories
            .iter()
            .enumerate()
            .map(|(i, mem)| {
                let status = if mem.is_correct { "正确" } else { "错误" };
                let mut text = format!(
                    "[记录 {}]\n- 问题: {}\n- 方法: {}\n- 答案: {}\n- 结果: {}",
                    i + 1,
                    mem.task_summary,
                    mem.approach,
                    mem.final_answer,
                    status
                );
                if !mem.is_correct
                    && let Some(analysis) = &mem.error_analysis
                {
                    text.push_str(&format!("\n- 错误分析: {analysis}"));
                }
                text
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[async_trait::async_trait]
impl Tool for RecallMemoryTool {
    fn name(&self) -> &str {
        "recall_memory"
    }

    fn description(&self) -> &str {
        "搜索过往的解题记录,用于检查是否遇到过类似的问题"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "要搜索的问题描述" }
            },
            "required": ["query"]
        })
    }

    fn requires_confirmation(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: RecallArgs = serde_json::from_str(args_json).map_err(|e| {
            anyhow::anyhow!("invalid arguments, expected {{\"query\": string}}: {e}")
        })?;

        let query = args.query.trim();
        if query.is_empty() {
            return Err(anyhow::anyhow!("argument `query` must not be empty"));
        }

        let memories = self.memory_manager.search(query, RECALL_TOP_K).await?;
        if memories.is_empty() {
            return Ok("没有找到相似的解题记录。".to_string());
        }

        Ok(Self::format_memories(&memories))
    }
}

#[cfg(test)]
mod recall_memory_tests {
    use super::*;

    fn memory(is_correct: bool, error_analysis: Option<&str>) -> TaskMemory {
        TaskMemory {
            task_summary: "圆的面积".to_string(),
            approach: "π r²".to_string(),
            final_answer: "78.5".to_string(),
            is_correct,
            error_analysis: error_analysis.map(str::to_string),
        }
    }

    #[test]
    fn incorrect_record_includes_error_analysis() {
        let text = RecallMemoryTool::format_memories(&[memory(false, Some("把直径当成了半径"))]);

        assert!(text.contains("结果: 错误"));
        assert!(text.contains("错误分析: 把直径当成了半径"));
    }

    #[test]
    fn correct_record_omits_error_analysis() {
        let text = RecallMemoryTool::format_memories(&[memory(true, None)]);

        assert!(text.contains("结果: 正确"));
        assert!(!text.contains("错误分析"));
    }

    #[test]
    fn records_are_numbered_and_separated() {
        let text = RecallMemoryTool::format_memories(&[memory(true, None), memory(true, None)]);

        assert!(text.contains("[记录 1]"));
        assert!(text.contains("[记录 2]"));
    }

    #[test]
    fn args_without_query_fail_to_parse() {
        assert!(serde_json::from_str::<RecallArgs>("{}").is_err());
    }
}
