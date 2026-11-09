use std::sync::Arc;

use crate::{
    agent::model::{AgentResult, AgentStatus},
    agent::context::ExecutionContext,
    multi_agent::node::AgentNode,
};

/// (最新结果, 从 1 开始的轮数) -> 是否停止
pub type StopCondition =
    Arc<dyn Fn(&AgentResult<serde_json::Value>, usize) -> bool + Send + Sync>;

pub struct LoopWorkflow {
    name: String,
    agents: Vec<Arc<dyn AgentNode>>,
    stop_condition: Option<StopCondition>,
    max_iterations: usize,
}

impl LoopWorkflow {
    /// `max_iterations` 是必填的：循环必须有上限。
    pub fn new(agents: Vec<Arc<dyn AgentNode>>, max_iterations: usize) -> Self {
        Self {
            name: "loop_workflow".to_owned(),
            agents,
            stop_condition: None,
            max_iterations,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// 没有停止条件时，循环就执行 max_iterations 次。
    pub fn with_stop_condition<F>(mut self, condition: F) -> Self
    where
        F: Fn(&AgentResult<serde_json::Value>, usize) -> bool + Send + Sync + 'static,
    {
        self.stop_condition = Some(Arc::new(condition));
        self
    }
}

#[async_trait::async_trait]
impl AgentNode for LoopWorkflow {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(
        &self,
        user_input: Option<String>,
        mut context: ExecutionContext,
    ) -> anyhow::Result<AgentResult<serde_json::Value>> {
        if self.agents.is_empty() {
            anyhow::bail!("LoopWorkflow '{}' has no agents", self.name);
        }
        if self.max_iterations == 0 {
            anyhow::bail!("LoopWorkflow '{}' needs max_iterations > 0", self.name);
        }

        // 只有第一轮的第一个节点拿到用户输入。
        let mut input = user_input;

        for iteration in 1..=self.max_iterations {
            tracing::info!(
                "loop '{}': iteration {}/{}",
                self.name,
                iteration,
                self.max_iterations
            );
            let mut last_output = None;

            for agent in &self.agents {
                let AgentResult { output, context: next, status } =
                    agent.execute(input.take(), context).await?;

                // 等待确认或出错：立刻停止并原样返回。
                if status != AgentStatus::Complete {
                    return Ok(AgentResult::new(output, next, status));
                }

                context = next;
                last_output = output;
            }

            let result = AgentResult::new(last_output, context, AgentStatus::Complete);

            let stop = self
                .stop_condition
                .as_ref()
                .is_some_and(|condition| condition(&result, iteration));
            if stop {
                tracing::info!("loop '{}': stop condition met at iteration {}", self.name, iteration);
                return Ok(result);
            }
            if iteration == self.max_iterations {
                tracing::warn!(
                    "loop '{}': reached max_iterations ({}) without meeting the stop condition",
                    self.name,
                    self.max_iterations
                );
                return Ok(result);
            }

            // 还没结束：把上下文取回来，进入下一轮。
            context = result.context;
        }

        unreachable!("max_iterations is checked to be > 0, so the loop always returns")
    }
}
