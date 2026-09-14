use std::sync::Arc;

use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionToolChoiceOption, ChatCompletionTools,
    CreateChatCompletionRequestArgs, ToolChoiceOptions,
};
use backon::{ExponentialBuilder, Retryable};

use crate::{
    agent::{
        callback::{AfterToolCallback, BeforeToolCallback},
        confirmation::{PendingToolCall, ToolConfirmation},
        llm_request::{final_answer_tool_definition, BeforeLlmCallback},
    },
    session::manager::{InMemorySessionManager, SessionManager},
    tools::ToolBox,
};

use super::{
    context::ExecutionContext,
    event::{ContentItem, Event, ToolResultStatus},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStatus {
    Complete,
    PendingConfirmation,
}

#[derive(Debug)]
pub struct AgentResult {
    pub output: Option<String>,
    pub context: ExecutionContext,
    pub status: AgentStatus,
    pub pending_tool_calls: Vec<PendingToolCall>,
}

#[derive(Debug)]
pub struct StructuredAgentResult<T> {
    pub output: T,
    pub context: ExecutionContext,
    pub status: AgentStatus,
}

pub struct Agent {
    pub(super) model: String,
    pub(super) instructions: Option<String>,
    pub(super) toolbox: Arc<ToolBox>,
    pub(super) max_steps: u32,
    pub(super) before_tool_callbacks: Vec<Arc<dyn BeforeToolCallback>>,
    pub(super) after_tool_callbacks: Vec<Arc<dyn AfterToolCallback>>,
    pub(super) before_llm_callbacks: Vec<Arc<dyn BeforeLlmCallback>>,
    pub(super) session_manager: Box<dyn SessionManager>,
}

impl Agent {
    pub fn new(
        model: impl Into<String>,
        instructions: Option<impl Into<String>>,
        toolbox: Arc<ToolBox>,
    ) -> Self {
        Self {
            model: model.into(),
            instructions: instructions.map(Into::into),
            toolbox,
            max_steps: 10,
            before_tool_callbacks: Vec::new(),
            after_tool_callbacks: Vec::new(),
            before_llm_callbacks: Vec::new(),
            session_manager: Box::new(InMemorySessionManager::new()),
        }
    }

    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn with_before_tool_callback(mut self, callback: Arc<dyn BeforeToolCallback>) -> Self {
        self.before_tool_callbacks.push(callback);
        self
    }

    pub fn with_after_tool_callback(mut self, callback: Arc<dyn AfterToolCallback>) -> Self {
        self.after_tool_callbacks.push(callback);
        self
    }

    pub fn with_before_llm_callback(mut self, callback: Arc<dyn BeforeLlmCallback>) -> Self {
        self.before_llm_callbacks.push(callback);
        self
    }

    pub fn with_session_manager(mut self, session_manager: impl SessionManager + 'static) -> Self {
        self.session_manager = Box::new(session_manager);
        self
    }

    /// Runs the agent loop until it produces a final answer or pauses on a
    /// tool call that requires human confirmation. Pass the previous
    /// `PendingConfirmation` result's tool calls back in as
    /// `tool_confirmations`, together with `user_input: None`, to resume.
    pub async fn run(
        &self,
        user_input: Option<&str>,
        session_id: &str,
        tool_confirmations: Option<Vec<ToolConfirmation>>,
    ) -> anyhow::Result<AgentResult> {
        let session = self.session_manager.get_or_create(session_id, None).await?;
        let mut context = ExecutionContext::new(session);

        if let Some(input) = user_input {
            context.add_event(Event::new(
                context.execution_id.clone(),
                "user",
                vec![ContentItem::Message {
                    role: "user".to_string(),
                    content: input.to_string(),
                }],
            ));
        }

        if let Some(confirmations) = tool_confirmations {
            let value = serde_json::to_value(&confirmations)?;
            context
                .state_mut()
                .insert("tool_confirmations".to_string(), value);
        }

        let client = async_openai::Client::new();

        let tool_definitions: Vec<ChatCompletionTools> = self
            .toolbox
            .values()
            .filter_map(|t| match t.definition() {
                Ok(def) => Some(def),
                Err(e) => {
                    tracing::warn!("Skip tool {}, failed to get its definition: {e}", t.name());
                    None
                }
            })
            .collect();

        loop {
            if context.current_step >= self.max_steps {
                anyhow::bail!(
                    "Agent exceeded the maximum of {} steps without a final answer",
                    self.max_steps
                );
            }

            let llm_request = self.prepare_llm_request(&mut context).await?;
            let messages = self.build_messages(&llm_request)?;

            let request = CreateChatCompletionRequestArgs::default()
                .model(self.model.clone())
                .messages(messages)
                .tools(tool_definitions.clone())
                .max_tokens(2048u32)
                .build()?;

            let response = (|| async { client.chat().create(request.clone()).await })
                .retry(ExponentialBuilder::default().with_max_times(3))
                .await?;

            if let Some(usage) = &response.usage {
                context.usage.add(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                );
            }

            let message = response
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No choices in response"))?
                .message;

            if let Some(tool_calls) = message.tool_calls {
                self.record_tool_calls(&mut context, &tool_calls);
                self.execute_tool_calls(&mut context, &tool_calls).await;

                if let Some(raw_pending) = context.state_mut().get("pending_tool_calls").cloned() {
                    let pending_tool_calls: Vec<PendingToolCall> =
                        serde_json::from_value(raw_pending)?;
                    self.session_manager.save(context.session.clone()).await?;
                    return Ok(AgentResult {
                        output: None,
                        context,
                        status: AgentStatus::PendingConfirmation,
                        pending_tool_calls,
                    });
                }
            } else {
                let content = message
                    .content
                    .ok_or_else(|| anyhow::anyhow!("No content in final response"))?;

                context.add_event(Event::new(
                    context.execution_id.clone(),
                    "agent",
                    vec![ContentItem::Message {
                        role: "assistant".to_string(),
                        content: content.clone(),
                    }],
                ));
                context.final_result = Some(content.clone());
                self.session_manager.save(context.session.clone()).await?;
                return Ok(AgentResult {
                    output: Some(content),
                    context,
                    status: AgentStatus::Complete,
                    pending_tool_calls: Vec::new(),
                });
            }

            context.increment_step();
        }
    }

    pub async fn run_structured<T>(
        &self,
        user_input: &str,
        session_id: &str,
    ) -> anyhow::Result<StructuredAgentResult<T>>
    where
        T: schemars::JsonSchema + serde::de::DeserializeOwned,
    {
        let session = self.session_manager.get_or_create(session_id, None).await?;
        let mut context = ExecutionContext::new(session);

        context.add_event(Event::new(
            context.execution_id.clone(),
            "user",
            vec![ContentItem::Message {
                role: "user".to_string(),
                content: user_input.to_string(),
            }],
        ));

        let client = async_openai::Client::new();

        let mut tool_definitions: Vec<ChatCompletionTools> = self
            .toolbox
            .values()
            .filter_map(|t| match t.definition() {
                Ok(def) => Some(def),
                Err(e) => {
                    tracing::warn!("Skip tool {}, failed to get its definition: {e}", t.name());
                    None
                }
            })
            .collect();
        tool_definitions.push(final_answer_tool_definition::<T>()?);

        loop {
            if context.current_step >= self.max_steps {
                anyhow::bail!(
                    "Agent exceeded the maximum of {} steps without a final answer",
                    self.max_steps
                );
            }

            let llm_request = self.prepare_llm_request(&mut context).await?;
            let messages = self.build_messages(&llm_request)?;

            let request = CreateChatCompletionRequestArgs::default()
                .model(self.model.clone())
                .messages(messages)
                .tools(tool_definitions.clone())
                .tool_choice(ChatCompletionToolChoiceOption::Mode(
                    ToolChoiceOptions::Required,
                ))
                .max_tokens(2048u32)
                .build()?;

            let response = (|| async { client.chat().create(request.clone()).await })
                .retry(ExponentialBuilder::default().with_max_times(3))
                .await?;

            if let Some(usage) = &response.usage {
                context.usage.add(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                );
            }

            let message = response
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No choices in response"))?
                .message;

            let tool_calls = message.tool_calls.ok_or_else(|| {
                anyhow::anyhow!("Model returned no tool call despite tool_choice = required")
            })?;

            self.record_tool_calls(&mut context, &tool_calls);

            let final_call = tool_calls.iter().find_map(|tool_call| match tool_call {
                ChatCompletionMessageToolCalls::Function(f)
                    if f.function.name == "final_answer" =>
                {
                    Some(f)
                }
                _ => None,
            });

            if let Some(final_call) = final_call {
                let raw_arguments = final_call.function.arguments.clone();
                let parsed: T = serde_json::from_str(&raw_arguments)?;

                context.add_event(Event::new(
                    context.execution_id.clone(),
                    "tool",
                    vec![ContentItem::ToolResult {
                        tool_call_id: final_call.id.clone(),
                        name: "final_answer".to_string(),
                        status: ToolResultStatus::Success,
                        content: raw_arguments.clone(),
                    }],
                ));
                context.final_result = Some(raw_arguments);
                self.session_manager.save(context.session.clone()).await?;
                return Ok(StructuredAgentResult {
                    output: parsed,
                    context,
                    status: AgentStatus::Complete,
                });
            }

            self.execute_tool_calls(&mut context, &tool_calls).await;
            context.increment_step();
        }
    }
}
