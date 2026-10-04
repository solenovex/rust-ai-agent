use std::{collections::HashMap, fmt, sync::Arc};

use async_openai::types::chat::ChatCompletionTools;

use crate::{
    agent::{
        callback::{
            after_tool_callback::AfterToolCallback, before_llm_callback::BeforeLlmCallback,
            before_tool_callback::BeforeToolCallback,
        },
        context::ExecutionContext,
        event::{AgentEvent, ContentItem, ToolResult, ToolResultStatus},
        tool_confirmation::ToolConfirmation,
    },
    constant::{AGENT_NAME, FINAL_ANSWER_TOOL_NAME},
    llm::llm_request::{LlmRequest, ToolChoice},
    session::session_manager::{InMemorySessionManager, SessionManager},
    tool::ToolBox,
};

pub struct Agent {
    pub name: String,
    pub model: String,
    pub instructions: Vec<String>,
    pub toolbox: Arc<ToolBox>,
    pub max_steps: usize,
    pub before_tool_callbacks: Vec<Arc<dyn BeforeToolCallback>>,
    pub after_tool_callbacks: Vec<Arc<dyn AfterToolCallback>>,
    pub before_llm_callbacks: Vec<Arc<dyn BeforeLlmCallback>>,
    pub session_manager: Box<dyn SessionManager>,
}

impl Agent {
    pub fn new(model: String, instructions: Vec<String>, toolbox: Arc<ToolBox>) -> Self {
        Self {
            model,
            instructions,
            toolbox,
            max_steps: 10,
            name: AGENT_NAME.to_owned(),
            before_tool_callbacks: Vec::new(),
            after_tool_callbacks: Vec::new(),
            before_llm_callbacks: Vec::new(),
            session_manager: Box::new(InMemorySessionManager::new()),
        }
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn with_before_tool_callbacks(mut self, callback: Arc<dyn BeforeToolCallback>) -> Self {
        self.before_tool_callbacks.push(callback);
        self
    }

    pub fn with_after_tool_callbacks(mut self, callback: Arc<dyn AfterToolCallback>) -> Self {
        self.after_tool_callbacks.push(callback);
        self
    }

    pub fn with_before_llm_callbacks(mut self, callback: Arc<dyn BeforeLlmCallback>) -> Self {
        self.before_llm_callbacks.push(callback);
        self
    }

    pub fn with_session_manager(mut self, session_manager: impl SessionManager + 'static) -> Self {
        self.session_manager = Box::new(session_manager);
        self
    }

    pub async fn prepare_llm_request(
        &self,
        context: &mut ExecutionContext,
    ) -> anyhow::Result<LlmRequest> {
        if !context.session.pending_tool_calls.is_empty()
            && context.session.state.contains_key("tool_confirmations")
        {
            let confirmation_results = self.process_confirmations(context).await?;

            if !confirmation_results.is_empty() {
                context.session.add_event(AgentEvent::new(
                    context.execution_id.clone(),
                    "tool",
                    confirmation_results,
                ));
            }

            context.session.pending_tool_calls.clear();
            context.session.state.remove("tool_confirmations");
        }

        let tool_choice = if self.has_final_answer_tool() {
            ToolChoice::Required
        } else {
            ToolChoice::Auto
        };

        let content_items = context
            .session
            .events
            .iter()
            .flat_map(|ev| ev.content.iter().cloned())
            .collect();
        let mut llm_request =
            LlmRequest::new(self.instructions.clone(), content_items).with_tool_choice(tool_choice);

        for before_llm_callback in &self.before_llm_callbacks {
            before_llm_callback
                .before_llm_call(&mut context.session, &mut llm_request)
                .await
        }

        Ok(llm_request)
    }

    async fn process_confirmations(
        &self,
        context: &ExecutionContext,
    ) -> anyhow::Result<Vec<ContentItem>> {
        let pending_calls = &context.session.pending_tool_calls;
        let confirmations: Vec<ToolConfirmation> = serde_json::from_value(
            context
                .session
                .state
                .get("tool_confirmations")
                .unwrap()
                .clone(),
        )?;

        let confirmation_map: HashMap<String, ToolConfirmation> = confirmations
            .into_iter()
            .map(|c| (c.tool_call_id.clone(), c))
            .collect();

        let mut results = Vec::with_capacity(pending_calls.len());

        for pending_call in pending_calls {
            let tool_call_id = pending_call.tool_call.id.clone();
            let name = pending_call.tool_call.function.name.clone();
            let confirmation = confirmation_map.get(&tool_call_id);

            let (status, content) = match confirmation {
                Some(c) if c.approved => {
                    let mut arguments: serde_json::Map<String, serde_json::Value> =
                        serde_json::from_str(&pending_call.tool_call.function.arguments)?;
                    if let Some(overrides) = &c.modified_arguments {
                        arguments.extend(overrides.clone());
                    }
                    let args_json = serde_json::Value::Object(arguments).to_string();

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

            results.push(ContentItem::ToolResult(ToolResult::new(
                tool_call_id,
                name,
                status,
                content,
            )));
        }

        Ok(results)
    }

    pub fn tool_definitions(&self) -> Vec<ChatCompletionTools> {
        self.toolbox
            .iter()
            .filter_map(|t| t.1.definition().ok())
            .collect()
    }

    pub fn has_final_answer_tool(&self) -> bool {
        self.toolbox.contains_key(FINAL_ANSWER_TOOL_NAME)
    }
}

#[derive(Debug)]
pub struct AgentResult<T> {
    pub output: Option<T>,
    pub context: ExecutionContext,
    pub status: AgentStatus,
}

impl<T> AgentResult<T> {
    pub fn new(output: Option<T>, context: ExecutionContext, status: AgentStatus) -> Self {
        Self {
            output,
            context,
            status,
        }
    }
}

impl<T: fmt::Display> fmt::Display for AgentResult<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.output {
            Some(output) => write!(f, "{output}"),
            None => write!(f, "No output"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStatus {
    Complete,
    PendingConfirmation,
    Error,
}
