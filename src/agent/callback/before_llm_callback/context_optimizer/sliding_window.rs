use std::collections::HashSet;

use crate::{
    agent::{
        callback::before_llm_callback::context_optimizer::{ContextOptimizer, count_token},
        context::ExecutionContext,
        event::ContentItem,
    },
    llm::llm_request::LlmRequest,
};

impl ContextOptimizer {
    pub fn slide_window(
        &self,
        context: &ExecutionContext,
        llm_request: &mut LlmRequest,
        window_size: usize,
    ) {
        let token_count_before = count_token(&self.model, llm_request);
        if token_count_before < self.token_threshold {
            return;
        }

        let contents: &[ContentItem] = &llm_request.contents;
        if contents.len() <= window_size {
            return;
        }

        let user_message_idx = contents.iter().enumerate().find_map(|(i, c)| {
            if let ContentItem::Message(msg) = c
                && msg.role == "user"
            {
                Some(i)
            } else {
                None
            }
        });

        if let Some(user_message_idx) = user_message_idx {
            let preserved = &contents[..=user_message_idx];
            let mut remaining = &contents[user_message_idx + 1..];
            if remaining.len() > window_size {
                remaining = &remaining[remaining.len() - window_size..];
            }

            let tool_call_ids: HashSet<&str> = remaining
                .iter()
                .filter_map(|ci| {
                    if let ContentItem::ToolCall(tc) = ci {
                        Some(tc.tool_call_id.as_str())
                    } else {
                        None
                    }
                })
                .collect();

            let orphan_tool_result_ids: Vec<&str> = remaining
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
                context.session.events.iter().flat_map(|ev| ev.content.iter()).filter(|&ci| {
                    matches!(ci, ContentItem::ToolCall(tc) if orphan_tool_result_ids.contains(&tc.tool_call_id.as_str()))
                }).cloned().collect();

            llm_request.contents = preserved
                .iter()
                .chain(tool_calls.iter())
                .chain(remaining.iter())
                .cloned()
                .collect();

            tracing::info!(
                "SlidingWindow: {token_count_before} tokens >= threshold {}, trimming to last {} items",
                self.token_threshold,
                window_size
            );
        }
    }
}
