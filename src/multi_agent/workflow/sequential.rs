use std::sync::Arc;

use crate::{agent::{context::ExecutionContext, model::{AgentResult, AgentStatus}}, multi_agent::node::AgentNode};

pub struct SequentialWorkflow {
    name: String,
    agents: Vec<Arc<dyn AgentNode>>
}

impl SequentialWorkflow {
    pub fn new(agents: Vec<Arc<dyn AgentNode>>) -> Self {
        Self {
            name: "sequential_workflow".to_owned(),
            agents,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

#[async_trait::async_trait]
impl AgentNode for SequentialWorkflow {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, user_input: Option<String>, mut context: ExecutionContext) -> anyhow::Result<AgentResult<serde_json::Value>> {
        if self.agents.is_empty() {
            anyhow::bail!("SequentialWorkflow '{}' has no agents", self.name);
        }

        let mut input = user_input;
        let mut last_output = None;
        let mut last_status = AgentStatus::Complete;

        for agent in &self.agents {
            // First agent gets the user input; `take()` leaves None for the rest.
            let AgentResult {output, context: next, status} = agent.execute(input.take(), context).await?;

            // Pending confirmation or error: stop and hand it straight back.
            if status != AgentStatus::Complete {
                return Ok(AgentResult::new(output, next, status));
            }

            context = next;
            last_output = output;
            last_status = status;
        }

        Ok(AgentResult::new(last_output, context, last_status))
    }
}
