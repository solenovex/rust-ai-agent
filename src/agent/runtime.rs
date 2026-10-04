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
    pub async fn run<T: DeserializeOwned>(
        &self,
        user_prompt: &str,
        session_id: &str,
        tool_confirmations: Option<Vec<ToolConfirmation>>,
    ) -> anyhow::Result<AgentResult<T>> {
        let session = self.session_manager.get_or_create(session_id, None).await?;
        let mut context = ExecutionContext::new(session);

        if !user_prompt.is_empty() {
            let user_event = AgentEvent::new(
                context.execution_id.clone(),
                "user",
                vec![ContentItem::Message(Message::new(
                    "user".to_owned(),
                    user_prompt.to_owned(),
                ))],
            );
            context.session.add_event(user_event);
        }

        if let Some(confirmations) = tool_confirmations {
            let value = serde_json::to_value(&confirmations)?;
            context
                .session
                .state
                .insert("tool_confirmations".to_string(), value);
        }

        let client = async_openai::Client::new();
        println!("You> {user_prompt}");
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
                        serde_json::from_value(value)?,
                        context,
                        AgentStatus::Complete,
                    ));
                }
            }
        }
    }
}
