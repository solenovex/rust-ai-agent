use serde::de::DeserializeOwned;

use crate::agent::{
    context::ExecutionContext,
    event::{AgentEvent, ContentItem, Message},
    model::{Agent, AgentResult, AgentStatus},
    runtime::step::StepResult,
    tool_confirmation::ToolConfirmation,
};

pub mod act;
pub mod step;
pub mod think;

impl Agent {
    /// 在已有的上下文上运行循环。工作流内部会调用它。
    pub async fn run_in_context(
        &self,
        mut context: ExecutionContext,
    ) -> anyhow::Result<AgentResult<serde_json::Value>> {
        let client = async_openai::Client::new();
        loop {
            if context.current_step >= self.max_steps {
                anyhow::bail!(
                    "Agent exceeded the maximum of {} steps without a final answer",
                    self.max_steps
                );
            }

            match self.step(&client, &mut context).await? {
                StepResult::Continue => {
                    context.increment_step();
                    if !context.session.pending_tool_calls.is_empty() {
                        self.session_manager.save(context.session.clone()).await?;
                        return Ok(AgentResult::new(
                            None,
                            context,
                            AgentStatus::PendingConfirmation,
                        ));
                    }
                }
                StepResult::Final(final_result) => {
                    let value = if self.has_final_answer_tool() {
                        serde_json::from_str::<serde_json::Value>(&final_result)?
                    } else {
                        serde_json::Value::String(final_result)
                    };
                    self.session_manager.save(context.session.clone()).await?;
                    return Ok(AgentResult::new(
                        Some(value),
                        context,
                        AgentStatus::Complete,
                    ));
                }
            }
        }
    }

    /// 原来的 run 现在是薄包装：创建上下文 → 加入用户消息 → 调用 run_in_context。
    pub async fn run<T: DeserializeOwned>(
        &self,
        user_prompt: &str,
        session_id: &str,
        tool_confirmations: Option<Vec<ToolConfirmation>>,
    ) -> anyhow::Result<AgentResult<T>> {
        let session = self.session_manager.get_or_create(session_id, None).await?;
        let mut context = ExecutionContext::new(session);

        push_user_message(&mut context, user_prompt);

        if let Some(confirmations) = tool_confirmations {
            let value = serde_json::to_value(&confirmations)?;
            context
                .session
                .state
                .insert("tool_confirmations".to_string(), value);
        }

        println!("You> {user_prompt}");

        let result = self.run_in_context(context).await?;

        let output = match result.output {
            Some(value) => serde_json::from_value::<Option<T>>(value)?,
            None => None,
        };

        Ok(AgentResult::new(output, result.context, result.status))
    }
}

/// 把用户消息加入上下文。空字符串会被忽略。
pub fn push_user_message(context: &mut ExecutionContext, text: &str) {
    if !text.is_empty() {
        let user_event = AgentEvent::new(
            context.execution_id.clone(),
            "user",
            vec![ContentItem::Message(Message::new(
                "user".to_owned(),
                text.to_owned(),
            ))],
        );
        context.session.add_event(user_event);
    }
}
