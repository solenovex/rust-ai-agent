use std::sync::Arc;

use futures::future::try_join_all;

use crate::{
    agent::{
        context::ExecutionContext,
        model::{AgentResult, AgentStatus},
        runtime::push_user_message,
    },
    multi_agent::node::AgentNode,
};

pub struct ParallelWorkflow {
    name: String,
    agents: Vec<Arc<dyn AgentNode>>,
}

impl ParallelWorkflow {
    pub fn new(agents: Vec<Arc<dyn AgentNode>>) -> Self {
        Self {
            name: "parallel_workflow".to_owned(),
            agents,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

#[async_trait::async_trait]
impl AgentNode for ParallelWorkflow {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(
        &self,
        user_input: Option<String>,
        mut context: ExecutionContext,
    ) -> anyhow::Result<AgentResult<serde_json::Value>> {
        if self.agents.is_empty() {
            anyhow::bail!("ParallelWorkflow '{}' has no agents", self.name);
        }

        // 用户消息在分叉之前只加一次，分支里不会重复。
        if let Some(input) = user_input {
            push_user_message(&mut context, &input);
        }
        let base_len = context.session.events.len();

        // 每个分支拿到一份副本，并且"继续"（不加新消息）。
        let results = try_join_all(
            self.agents
                .iter()
                .map(|agent| agent.execute(None, context.clone())),
        )
        .await?;

        let mut sections = Vec::with_capacity(results.len());
        for (agent, result) in self.agents.iter().zip(results) {
            if result.status != AgentStatus::Complete {
                anyhow::bail!(
                    "branch '{}' ended with status {:?}; confirmation inside parallel branches is not supported yet",
                    agent.name(),
                    result.status
                );
            }

            // 只追加这个分支在公共基础之后新增的事件。
            context
                .session
                .events
                .extend(result.context.session.events.into_iter().skip(base_len));

            let text = match result.output {
                Some(serde_json::Value::String(s)) => s,
                Some(other) => other.to_string(),
                None => String::new(),
            };
            sections.push(format!("[{}]\n{}", agent.name(), text));
        }

        Ok(AgentResult::new(
            Some(serde_json::Value::String(sections.join("\n\n"))),
            context,
            AgentStatus::Complete,
        ))
    }
}
