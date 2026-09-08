use anyhow::Ok;
use async_openai::types::chat::{ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs};
use serde_json::Value;

use crate::{agent::{ContentItem, ExecutionContext, llm_request::LlmRequest}, callback::context_optimizer::find_safe_start};

const SUMMARIZATION_PROMPT: &str = "You are summarizing an AI agent's work-in-progress \
history. Given the execution history below, write a short, structured summary covering: \
1) key findings so far, 2) tools that were called, 3) what remains to be done. \
Be concise — a few sentences is enough.\n\nExecution history:\n{history}";

pub struct Summarization {
    pub model: String,
    pub keep_recent: usize,
}

impl Summarization {
    pub async fn apply(
        &self,
        context: &mut ExecutionContext,
        request: &mut LlmRequest,
    ) -> anyhow::Result<()> {
        if request.contents.len() <= self.keep_recent {
            return Ok(());
        }

        let user_idx = request.contents.iter().position(|item|matches!(item, 
        ContentItem::Message { role, .. } if role == "user"));
        let Some(user_idx) = user_idx else {
            return Ok(());
        };

        let last_summary_idx = context.state.get("last_summary_idx").and_then(Value::as_u64)
        .map(|v|v as usize).unwrap_or(user_idx);

        let summary_start = last_summary_idx + 1;
        let summary_end = find_safe_start(&request.contents, request.contents.len() - self.keep_recent);
        if summary_end <= summary_start {
            return Ok(());
        }

        let to_summarize = &request.contents[summary_start..summary_end];
        let history_text = format_history(to_summarize);
        let new_summary_chunk = self.generate_summary(&history_text).await?;

        let existing_summary = context.state.get("context_summary").and_then(Value::as_str)
        .map(str::to_string);
        let full_summary = match existing_summary {
            Some(prev) => format!("{prev}\n\n{new_summary_chunk}"),
            None => new_summary_chunk
        };
        request.append_instructions(format!("[Summary of earlier progress]\n{full_summary}"));

        request.contents.drain((user_idx + 1)..summary_end);
        context.state.insert("last_summary_idx".to_string(), Value::from((summary_end - 1) as u64));
        context.state.insert("context_summary".to_string(), Value::String(full_summary));

        Ok(())
    }

    async fn generate_summary(&self, history: &str) -> anyhow::Result<String> {
        let client = async_openai::Client::new();
        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model.clone())
            .messages(vec![
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(SUMMARIZATION_PROMPT.replace("{history}", history))
                    .build()?
                    .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content("Please summarize.")
                    .build()?
                    .into(),
            ])
            .max_tokens(512u32)
            .build()?;

        let response = client.chat().create(request).await?;
        Ok(response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .unwrap_or_default())
    }
}


fn format_history(items: &[ContentItem]) -> String {
    items
        .iter()
        .map(|item| match item {
            ContentItem::Message { role, content } => {
                let preview: String = content.chars().take(500).collect();
                format!("[{role}]: {preview}")
            }
            ContentItem::ToolCall {
                name, arguments, ..
            } => {
                format!("[tool call]: {name}({arguments})")
            }
            ContentItem::ToolResult { name, content, .. } => {
                let preview: String = content.chars().take(200).collect();
                format!("[tool result]: {name} -> {preview}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}
