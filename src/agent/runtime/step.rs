use anyhow::Ok;
use async_openai::config::OpenAIConfig;

use crate::{
    agent::{
        context::ExecutionContext,
        event::{AgentEvent, ContentItem, Message, ToolResultStatus},
        model::Agent,
    },
    constant::FINAL_ANSWER_TOOL_NAME,
};

impl Agent {
    pub async fn step(
        &self,
        client: &async_openai::Client<OpenAIConfig>,
        context: &mut ExecutionContext,
    ) -> anyhow::Result<StepResult> {
        let llm_request = self.prepare_llm_request(context).await?;
        let response = self.think(client, llm_request).await?;
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Model returned no choices"))?;
        let finish_reason = choice.finish_reason;
        let response_message = choice.message;

        if let Some(tool_calls) = response_message.tool_calls {
            self.act_all(&tool_calls, context).await?;
            if let Some(output) = self.final_answer_output(context) {
                return Ok(StepResult::Final(output));
            }
        } else if let Some(content) = response_message.content {
            println!(" AI> {content}");
            context.session.add_event(AgentEvent::new(
                context.execution_id.clone(),
                "agent",
                vec![ContentItem::Message(Message::new(
                    "assistant".to_owned(),
                    content.clone(),
                ))],
            ));
            if self.has_final_answer_tool() {
                return Ok(StepResult::Continue);
            }
            return Ok(StepResult::Final(content));
        } else {
            if let Some(refusal) = &response_message.refusal {
                tracing::warn!("Model refused: {refusal}");
            }
            anyhow::bail!(
                "Empty response from model: finish_reason={:?}, message={:?}",
                finish_reason,
                response_message
            );
        }
        Ok(StepResult::Continue)
    }

    pub fn final_answer_output(&self, context: &ExecutionContext) -> Option<String> {
        if !self.has_final_answer_tool() {
            return None;
        }
        context
            .session
            .events
            .iter()
            .flat_map(|ev| ev.content.iter())
            .rev()
            .find_map(|item| match item {
                ContentItem::ToolResult(r)
                    if r.name == FINAL_ANSWER_TOOL_NAME
                        && matches!(r.status, ToolResultStatus::Success) =>
                {
                    Some(r.content.clone())
                }
                _ => None,
            })
    }
}

pub enum StepResult {
    Continue,
    Final(String),
}
