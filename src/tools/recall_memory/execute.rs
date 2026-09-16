use std::sync::Arc;

use crate::memory::{manager::TaskMemoryManager, task_memory::TaskMemory};

pub struct RecallMemoryTool {
    pub memory_manager: Arc<TaskMemoryManager>,
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
                    && let Some(analysis) = &mem.error_analysis {
                        text.push_str(&format!("\n- 错误分析: {analysis}"));
                    }
                text
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}
