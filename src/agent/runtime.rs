use std::sync::Arc;

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
        callback::{AfterToolCallback, BeforeToolCallback, ToolCallView},
        llm_request::{BeforeLlmCallback, LlmRequest},
    },
    tools::ToolBox,
};

use super::{
    context::ExecutionContext,
    event::{ContentItem, Event, ToolResultStatus},
};

#[derive(Debug)]
pub struct AgentResult {
    pub output: String,
    pub context: ExecutionContext,
}

#[derive(Debug)]
pub struct StructuredAgentResult<T> {
    pub output: T,
    pub context: ExecutionContext,
}

pub struct Agent {
    model: String,
    instructions: Option<String>,
    toolbox: Arc<ToolBox>,
    max_steps: u32,
    before_tool_callbacks: Vec<Arc<dyn BeforeToolCallback>>,
    after_tool_callbacks: Vec<Arc<dyn AfterToolCallback>>,
    before_llm_callbacks: Vec<Arc<dyn BeforeLlmCallback>>,
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

    pub async fn run(&self, user_input: &str) -> anyhow::Result<AgentResult> {
        let mut context = ExecutionContext::new();

        context.add_event(Event::new(
            context.execution_id.clone(),
            "user",
            vec![ContentItem::Message {
                role: "user".to_string(),
                content: user_input.to_string(),
            }],
        ));

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

            let llm_request = self.prepare_llm_request(&mut context).await;
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
                return Ok(AgentResult {
                    output: content,
                    context,
                });
            }

            context.increment_step();
        }
    }

    pub async fn run_structured<T>(
        &self,
        user_input: &str,
    ) -> anyhow::Result<StructuredAgentResult<T>>
    where
        T: schemars::JsonSchema + serde::de::DeserializeOwned,
    {
        let mut context = ExecutionContext::new();

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

            let llm_request = self.prepare_llm_request(&mut context).await;
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

                return Ok(StructuredAgentResult {
                    output: parsed,
                    context,
                });
            }

            self.execute_tool_calls(&mut context, &tool_calls).await;
            context.increment_step();
        }
    }

    async fn prepare_llm_request(&self, context: &mut ExecutionContext) -> LlmRequest {
        let mut request = LlmRequest {
            instructions: Vec::new(),
            contents: context
                .events
                .iter()
                .flat_map(|ev| ev.content.iter().cloned())
                .collect(),
        };

        for cb in &self.before_llm_callbacks {
            cb.call(context, &mut request).await;
        }

        request
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
                call_items.push(ContentItem::ToolCall {
                    tool_call_id: function_call.id.clone(),
                    name: function_call.function.name.clone(),
                    arguments,
                });
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

        for tool_call in tool_calls {
            let ChatCompletionMessageToolCalls::Function(function_call) = tool_call else {
                continue;
            };
            let function_name = &function_call.function.name;
            let arguments = &function_call.function.arguments;

            tracing::info!("Tool call: {function_name}({arguments})");

            let arguments_value: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
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
                ContentItem::ToolCall {
                    tool_call_id,
                    name,
                    arguments,
                } => {
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
