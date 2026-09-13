use std::{collections::HashMap, sync::Arc};

use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionToolChoiceOption,
    ChatCompletionTools, CreateChatCompletionRequestArgs, FunctionCall, FunctionObjectArgs,
    ToolChoiceOptions,
};
use backon::{ExponentialBuilder, Retryable};
use serde_json::Value;

use crate::{
    agent::{
        callback::{AfterToolCallback, BeforeToolCallback, ToolCallView}, confirmation::{PendingToolCall, ToolConfirmation}, event::ToolCall, llm_request::{BeforeLlmCallback, LlmRequest},
    }, session::manager::{InMemorySessionManager, SessionManager}, tools::ToolBox,
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
    model: String,
    instructions: Option<String>,
    toolbox: Arc<ToolBox>,
    max_steps: u32,
    before_tool_callbacks: Vec<Arc<dyn BeforeToolCallback>>,
    after_tool_callbacks: Vec<Arc<dyn AfterToolCallback>>,
    before_llm_callbacks: Vec<Arc<dyn BeforeLlmCallback>>,
    session_manager: Box<dyn SessionManager>,
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
                    status: AgentStatus::Complete
                });
            }

            self.execute_tool_calls(&mut context, &tool_calls).await;
            context.increment_step();
        }
    }

    async fn prepare_llm_request(&self, context: &mut ExecutionContext) -> anyhow::Result<LlmRequest> {
        if let (Some(raw_pending), Some(raw_confirmations)) = (
            context.state_mut().remove("pending_tool_calls"),
            context.state_mut().remove("tool_confirmations"),
        ) {
            let confirmation_results = self
                .process_confirmations(context, raw_pending, raw_confirmations)
                .await?;

            if !confirmation_results.is_empty() {
                context.add_event(Event::new(
                    context.execution_id.clone(),
                    "tool",
                    confirmation_results,
                ));
            }
        }
        
        let mut request = LlmRequest {
            instructions: Vec::new(),
            contents: context
                .events()
                .iter()
                .flat_map(|ev| ev.content.iter().cloned())
                .collect(),
        };

        for cb in &self.before_llm_callbacks {
            cb.call(context, &mut request).await;
        }

        Ok(request)
    }

    async fn process_confirmations(
        &self,
        context: &ExecutionContext,
        raw_pending: Value,
        raw_confirmations: Value,
    ) -> anyhow::Result<Vec<ContentItem>> {
        let pending_calls: Vec<PendingToolCall> = serde_json::from_value(raw_pending)?;
        let confirmations: Vec<ToolConfirmation> = serde_json::from_value(raw_confirmations)?;

        let confirmation_map: HashMap<String, ToolConfirmation> = confirmations
            .into_iter()
            .map(|c| (c.tool_call_id.clone(), c))
            .collect();

        let mut results = Vec::with_capacity(pending_calls.len());

        for pending in pending_calls {
            let tool_call_id = pending.tool_call.tool_call_id;
            let name = pending.tool_call.name;
            let confirmation = confirmation_map.get(&tool_call_id);

            let (status, content) = match confirmation {
                Some(c) if c.approved => {
                    let mut arguments = pending
                        .tool_call
                        .arguments
                        .as_object()
                        .cloned()
                        .unwrap_or_default();
                    if let Some(overrides) = &c.modified_arguments {
                        arguments.extend(overrides.clone());
                    }
                    let args_json = Value::Object(arguments).to_string();

                    match self.toolbox.get(&name) {
                        Some(tool) => match tool.execute(&args_json, context).await {
                            Ok(result) => (ToolResultStatus::Success, result),
                            Err(err) => (
                                ToolResultStatus::Error,
                                format!("Tool execution error: {err}"),
                            ),
                        },
                        None => (
                            ToolResultStatus::Error,
                            format!("Tool execution error: unknown tool {name}"),
                        ),
                    }
                }
                Some(c) => {
                    let reason = c
                        .reason
                        .clone()
                        .unwrap_or_else(|| "Tool execution was rejected by user.".to_string());
                    (ToolResultStatus::Error, reason)
                }
                None => (
                    ToolResultStatus::Error,
                    "Tool execution was not approved.".to_string(),
                ),
            };

            results.push(ContentItem::ToolResult {
                tool_call_id,
                name,
                status,
                content,
            });
        }

        Ok(results)
    }


    fn record_tool_calls(
        &self,
        context: &mut ExecutionContext,
        tool_calls: &[ChatCompletionMessageToolCalls],
    ) {
        let mut call_items = Vec::new();
        for tool_call in tool_calls {
            if let ChatCompletionMessageToolCalls::Function(function_call) = tool_call {
                let arguments: serde_json::Value =
                    serde_json::from_str(&function_call.function.arguments)
                        .unwrap_or(serde_json::Value::Null);
                call_items.push(ContentItem::ToolCall(ToolCall {
                    tool_call_id: function_call.id.clone(),
                    name: function_call.function.name.clone(),
                    arguments,
                }));
            }
        }
        context.add_event(Event::new(
            context.execution_id.clone(),
            "agent",
            call_items,
        ));
    }

    async fn execute_tool_calls(
        &self,
        context: &mut ExecutionContext,
        tool_calls: &[ChatCompletionMessageToolCalls],
    ) {
        let mut result_items = Vec::new();
        let mut pending_tool_calls: Vec<PendingToolCall> = Vec::new();

        for tool_call in tool_calls {
            let ChatCompletionMessageToolCalls::Function(function_call) = tool_call else {
                continue;
            };
            let function_name = &function_call.function.name;
            let arguments = &function_call.function.arguments;

            tracing::info!("Tool call: {function_name}({arguments})");

            let arguments_value: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
            
            if let Some(tool) = self.toolbox.get(function_name)
                && tool.requires_confirmation() {
                    tracing::info!("Tool call requires confirmation: {function_name}");
                    pending_tool_calls.push(PendingToolCall {
                        tool_call: ToolCall {
                            tool_call_id: function_call.id.clone(),
                            name: function_name.clone(),
                            arguments: arguments_value.clone(),
                        },
                        confirmation_message: tool.get_confirmation_message(&arguments_value),
                    });
                    continue;
                }

            
            let view = ToolCallView {
                tool_call_id: &function_call.id,
                name: function_name,
                arguments: &arguments_value,
            };

            let mut short_circuited = None;
            for callback in &self.before_tool_callbacks {
                if let Some(result) = callback.call(context, view).await {
                    short_circuited = Some(result);
                    break;
                }
            }

            let (mut status, mut content) = match short_circuited {
                Some(result) => (ToolResultStatus::Success, result),
                None => match self.toolbox.get(function_name) {
                    Some(tool) => match tool.execute(arguments, context).await {
                        Ok(result) => {
                            tracing::info!("Tool result: {result}");
                            (ToolResultStatus::Success, result)
                        }
                        Err(err) => {
                            let msg = format!("Tool execution error: {err}");
                            tracing::error!("{msg}");
                            (ToolResultStatus::Error, msg)
                        }
                    },
                    None => {
                        let msg = format!("Tool execution error: unknown tool {function_name}");
                        tracing::error!("{msg}");
                        (ToolResultStatus::Error, msg)
                    }
                },
            };

            for callback in &self.after_tool_callbacks {
                if let Some((new_status, new_content)) = callback
                    .call(context, &function_call.id, function_name, status, &content)
                    .await
                {
                    status = new_status;
                    content = new_content;
                    break;
                }
            }

            result_items.push(ContentItem::ToolResult {
                tool_call_id: function_call.id.clone(),
                name: function_name.clone(),
                status,
                content,
            });
        }

        if !pending_tool_calls.is_empty() {
            let value = serde_json::to_value(&pending_tool_calls).unwrap_or(Value::Null);
            context
                .state_mut()
                .insert("pending_tool_calls".to_string(), value);
        }

        context.add_event(Event::new(
            context.execution_id.clone(),
            "tool",
            result_items,
        ));
    }

    fn build_messages(
        &self,
        request: &LlmRequest,
    ) -> anyhow::Result<Vec<ChatCompletionRequestMessage>> {
        let mut messages = Vec::new();

        if let Some(system) = &self.instructions {
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system.as_str())
                    .build()?
                    .into(),
            );
        }

        for extra_instruction in &request.instructions {
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(extra_instruction.as_str())
                    .build()?
                    .into(),
            );
        }

        for item in &request.contents {
            match item {
                ContentItem::Message { role, content } => {
                    let message: ChatCompletionRequestMessage = if role == "user" {
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(content.clone())
                            .build()?
                            .into()
                    } else {
                        ChatCompletionRequestAssistantMessageArgs::default()
                            .content(content.clone())
                            .build()?
                            .into()
                    };
                    messages.push(message);
                }
                ContentItem::ToolCall(ToolCall {
                    tool_call_id,
                    name,
                    arguments,
                }) => {
                    let tool_call =
                        ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                            id: tool_call_id.clone(),
                            function: FunctionCall {
                                name: name.clone(),
                                arguments: arguments.to_string(),
                            },
                        });

                    if let Some(ChatCompletionRequestMessage::Assistant(last)) = messages.last_mut()
                    {
                        last.tool_calls.get_or_insert_with(Vec::new).push(tool_call);
                    } else {
                        messages.push(
                            ChatCompletionRequestAssistantMessageArgs::default()
                                .tool_calls(vec![tool_call])
                                .build()?
                                .into(),
                        );
                    }
                }
                ContentItem::ToolResult {
                    tool_call_id,
                    content,
                    ..
                } => {
                    messages.push(
                        ChatCompletionRequestToolMessageArgs::default()
                            .tool_call_id(tool_call_id.clone())
                            .content(content.clone())
                            .build()?
                            .into(),
                    );
                }
            }
        }

        Ok(messages)
    }
}

fn final_answer_tool_definition<T: schemars::JsonSchema>() -> anyhow::Result<ChatCompletionTools> {
    let schema = schemars::schema_for!(T);
    let schema_json = serde_json::to_value(&schema)?;

    let function = FunctionObjectArgs::default()
        .name("final_answer")
        .description("Return the final structured answer matching the required schema.")
        .parameters(schema_json)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build final_answer tool definition: {e}"))?;

    Ok(ChatCompletionTools::Function(ChatCompletionTool {
        function,
    }))
}
