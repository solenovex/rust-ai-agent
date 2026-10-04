use std::collections::HashSet;

use async_openai::types::chat::{
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use serde_json::Value;

use crate::{
    agent::{
        callback::before_llm_callback::context_optimizer::{ContextOptimizer, count_token},
        event::ContentItem,
    },
    llm::llm_request::LlmRequest,
    session::model::Session,
};

const SUMMARIZATION_PROMPT: &str = "You are summarizing an AI agent's work-in-progress \
history. Given the execution history below, write a short, structured summary covering: \
1) key findings so far, 2) tools that were called, 3) what remains to be done. \
Be concise — a few sentences is enough.\n\nExecution history:\n{history}";

impl ContextOptimizer {
    pub async fn summarize(
        &self,
        session: &mut Session,
        llm_request: &mut LlmRequest,
    ) -> anyhow::Result<()> {
        let token_count_before = count_token(&self.model, llm_request);
        if token_count_before < self.token_threshold {
            return Ok(());
        }

        tracing::info!(
            "ContextOptimizer: {} tokens >= threshold {}, summarizing",
            token_count_before,
            self.token_threshold
        );

        if llm_request.contents.len() <= self.summarization_keep_recent {
            return Ok(());
        }

        let user_message_idx = llm_request
            .contents
            .iter()
            .enumerate()
            .find_map(|(i, c)| {
                if let ContentItem::Message(msg) = c
                    && msg.role == "user"
                {
                    Some(i)
                } else {
                    None
                }
            })
            .unwrap_or(0);

        let preserved_start = llm_request.contents[..=user_message_idx].to_vec();
        let remaining = &llm_request.contents[user_message_idx + 1..];
        if remaining.len() > self.summarization_keep_recent {
            let last_summary_idx = session
                .state
                .get("last_summary_idx")
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok());

            let summarization_end = llm_request.contents.len() - self.summarization_keep_recent;

            let summarization_start = last_summary_idx.map_or(user_message_idx + 1, |idx| idx + 1);
            if summarization_end <= summarization_start {
                return Ok(());
            }

            let to_summarize = &llm_request.contents[summarization_start..summarization_end];
            let history_text = format_history(to_summarize);

            tracing::info!(
                "ContextOptimizer: summarizing {} items (indexes {}..{})",
                to_summarize.len(),
                summarization_start,
                summarization_end
            );

            let new_summary_chunk = self.generate_summary(&history_text).await?;

            tracing::info!(
                "ContextOptimizer: summarization completed, generated {} chars",
                new_summary_chunk.len()
            );

            let existing_summary = session
                .state
                .get("context_summary")
                .and_then(Value::as_str)
                .map(str::to_string);
            let full_summary = match existing_summary {
                Some(prev) => format!("{prev}\n\n{new_summary_chunk}"),
                None => new_summary_chunk,
            };

            llm_request
                .append_instruction(format!("[Summary of earlier progress]\n{full_summary}"));

            session.state.insert(
                "last_summary_idx".to_owned(),
                Value::from((summarization_end - 1) as u64),
            );
            session.state.insert(
                "context_summary".to_string(),
                Value::String(full_summary.clone()),
            );

            tracing::info!(
                "ContextOptimizer: summary stored, last_summary_idx={}, total_summary_chars={}",
                summarization_end - 1,
                full_summary.len()
            );

            let preserved_end = llm_request.contents[summarization_end..].to_vec();
            let tool_call_ids: HashSet<&str> = preserved_end
                .iter()
                .filter_map(|ci| {
                    if let ContentItem::ToolCall(tc) = ci {
                        Some(tc.tool_call_id.as_str())
                    } else {
                        None
                    }
                })
                .collect();

            let orphan_tool_result_ids: Vec<&str> = preserved_end
                .iter()
                .filter_map(|ci| {
                    if let ContentItem::ToolResult(tr) = ci
                        && !tool_call_ids.contains(&tr.tool_call_id.as_str())
                    {
                        Some(tr.tool_call_id.as_str())
                    } else {
                        None
                    }
                })
                .collect();

            let tool_calls: Vec<ContentItem> =
                session.events.iter().flat_map(|ev| ev.content.iter()).filter(|&ci| {
                    matches!(ci, ContentItem::ToolCall(tc) if orphan_tool_result_ids.contains(&tc.tool_call_id.as_str()))
                }).cloned().collect();

            llm_request.contents = preserved_start
                .into_iter()
                .chain(tool_calls)
                .chain(preserved_end)
                .collect();
        }

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
            ContentItem::Message(msg) => {
                let preview: String = msg.content.chars().take(500).collect();
                format!("[{}]: {preview}", msg.role)
            }
            ContentItem::ToolCall(tc) => {
                format!("[tool call]: {}({})", tc.name, tc.arguments)
            }
            ContentItem::ToolResult(tr) => {
                let preview: String = tr.content.chars().take(200).collect();
                format!("[tool result]: {} -> {preview}", tr.name)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}
