use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};

use crate::{agent::context::ExecutionContext, planning::task::Task, tool::Tool};

pub struct CreateTasksTool;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CreateTasksArgs {
    pub tasks: Vec<Task>,
}

#[async_trait::async_trait]
impl Tool for CreateTasksTool {
    fn name(&self) -> &str {
        "create_tasks"
    }

    fn description(&self) -> &str {
        r"Create or update a task plan.

    WHEN TO USE:
    - Complex queries requiring multiple steps of research
    - Questions that need to combine information from different sources

    WHEN NOT TO USE:
    - Simple questions answerable with a single search
    - Tasks with obvious, straightforward procedures

    HOW TO USE:
    - Regenerate the entire task list with updated statuses
    - Mark completed tasks as 'completed'
    - Mark the next task to work on as 'in_progress'
    - Keep future tasks as 'pending'"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(CreateTasksArgs))
            .expect("Failed to serialize output schema")
    }

    fn requires_confirmation(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: CreateTasksArgs = serde_json::from_str(args_json)?;
        let result = create_tasks(&args.tasks);
        Ok(result)
    }
}

pub fn create_tasks(tasks: &[Task]) -> String {
    tasks
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod create_tasks_tool_tests {
    use super::*;
    use crate::planning::task::{Task, TaskStatus};

    #[test]
    fn create_tasks_renders_status_markers() {
        let tasks = vec![
            Task {
                content: "A".into(),
                status: TaskStatus::Completed,
            },
            Task {
                content: "B".into(),
                status: TaskStatus::InProgress,
            },
            Task {
                content: "C".into(),
                status: TaskStatus::Pending,
            },
        ];
        assert_eq!(create_tasks(&tasks), "[x] ~~A~~\n[>] **B**\n[ ] C");
    }

    #[test]
    fn args_reject_invalid_status() {
        let json = r#"{"tasks":[{"content":"A","status":"done"}]}"#;
        assert!(serde_json::from_str::<CreateTasksArgs>(json).is_err());
    }
}
