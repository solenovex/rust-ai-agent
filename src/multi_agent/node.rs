use crate::agent::{
    context::ExecutionContext,
    model::{Agent, AgentResult},
    runtime::push_user_message,
};

#[async_trait::async_trait]
pub trait AgentNode: Send + Sync {
    fn name(&self) -> &str;

    fn description(&self) -> &str {
        ""
    }

    /// `user_input = None` 表示"继续"：不再添加新的用户消息。
    async fn execute(
        &self,
        user_input: Option<String>,
        context: ExecutionContext,
    ) -> anyhow::Result<AgentResult<serde_json::Value>>;
}

#[async_trait::async_trait]
impl AgentNode for Agent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(
        &self,
        user_input: Option<String>,
        mut context: ExecutionContext,
    ) -> anyhow::Result<AgentResult<serde_json::Value>> {
        // 每个 Agent 都有自己的步数预算，即使上下文是继承来的。
        context.current_step = 0;
        if let Some(input) = user_input {
            push_user_message(&mut context, &input);
        }
        self.run_in_context(context).await
    }
}
