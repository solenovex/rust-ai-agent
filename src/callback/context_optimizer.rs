use std::collections::HashSet;

use crate::{agent::{ContentItem, ExecutionContext, llm_request::{BeforeLlmCallback, LlmRequest}}, callback::context_optimizer::{compaction::Compaction, summarization::Summarization}};

pub mod sliding_window;
pub mod compaction;
pub mod summarization;

pub struct ContextOptimizer{
    pub model: String,
    pub token_threshold: usize,
    pub enable_compaction: bool,
    pub compaction_keep_recent: usize,
    pub enable_summarization: bool,
    pub keep_recent: usize,
}

impl ContextOptimizer {
    pub fn new(model: impl Into<String>) -> Self {
        Self { 
            model: model.into(), 
            token_threshold: 20_000, 
            enable_compaction: true, 
            compaction_keep_recent: 4, 
            enable_summarization: true, 
            keep_recent: 5,
        }
    }
}

#[async_trait::async_trait]
impl BeforeLlmCallback for ContextOptimizer {
    async fn call(&self, context: &mut ExecutionContext, request: &mut LlmRequest) {
        let before: usize = count_tokens(&self.model, request);
        if before < self.token_threshold {
            return;
        }
        tracing::info!(
            "ContextOptimizer: {before} tokens >= threshold {}, optimizing", self.token_threshold
        );

        if self.enable_compaction {
            Compaction{
                keep_recent: self.compaction_keep_recent,
            }.apply(request);
            let after = count_tokens(&self.model, request);
            tracing::info!("ContextOptimizer: compaction brought it to {after} tokens");
            if after < self.token_threshold {
                return;
            }
        }

        if self.enable_summarization {
            let summarizer = Summarization{
                model: self.model.clone(),
                keep_recent: self.keep_recent,
            };
            match summarizer.apply(context, request).await{
                Ok(()) => {
                    let after = count_tokens(&self.model, request);
                    tracing::info!("ContextOptimizer: summarization brought it to {after} tokens");
                }
                Err(err) => tracing::warn!("Summarization skipped: {err}"),
            }
        }
    }
}

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