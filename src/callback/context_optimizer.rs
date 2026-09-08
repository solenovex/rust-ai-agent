use std::collections::HashSet;

use crate::agent::{ContentItem, llm_request::LlmRequest};

pub mod sliding_window;
pub mod compaction;
pub mod summarization;

pub fn count_tokens(_model: &str, request: &LlmRequest) -> usize {
    let bpe = tiktoken_rs::cl100k_base()
        .expect("tiktoken should always resolve to the cl100K_base encoding");

    let mut total = 0usize;

    for instruction in &request.instructions {
        total += 4 + bpe.encode_ordinary(instruction).len();
    }

    for item in &request.contents {
        total += 4;
        total += match item {
            ContentItem::Message { content, .. } => bpe.encode_ordinary(content).len(),
            ContentItem::ToolCall { name, arguments, .. } => {
                bpe.encode_ordinary(name).len() + bpe.encode_ordinary(&arguments.to_string()).len()
            },
            ContentItem::ToolResult { content, .. } => bpe.encode_ordinary(content).len(),
        }
    }

    total
}


fn find_safe_start(contents: &[ContentItem], mut start: usize) -> usize {
    loop {
        let call_ids: HashSet<&str> = contents[start..].iter().filter_map(|item| match item {
            ContentItem::ToolCall { tool_call_id, .. } => Some(tool_call_id.as_str()),
            _ => None,
        }).collect();

        let missing_call = contents[start..].iter().find_map(|item| match item {
            ContentItem::ToolResult { tool_call_id, .. } if ! call_ids.contains(&tool_call_id.as_str()) => {
                Some(tool_call_id)
            }
            _=> None,
        });

        let Some(missing_call) = missing_call else {
            break;
        };

        let Some(call_idx) = contents[..start].iter().position(|item| {
            matches!(item, 
            ContentItem::ToolCall { tool_call_id, .. } if tool_call_id == missing_call ) 
        }) else {
            break;
        };

        start = call_idx;
    }

    start
}