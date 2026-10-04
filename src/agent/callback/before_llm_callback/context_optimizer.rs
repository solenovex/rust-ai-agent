use crate::{
    agent::{
        callback::before_llm_callback::BeforeLlmCallback,
        event::{ContentItem, ToolCall},
    },
    llm::llm_request::LlmRequest,
    session::model::Session,
};

pub mod compaction;
pub mod sliding_window;
pub mod summarization;

pub struct ContextOptimizer {
    pub model: String,
    pub token_threshold: usize,
    pub enable_compaction: bool,
    pub enbale_summarization: bool,
    pub summarization_keep_recent: usize,
}

impl ContextOptimizer {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            token_threshold: 20_000,
            enable_compaction: false,
            enbale_summarization: true,
            summarization_keep_recent: 5,
        }
    }

    pub fn with_token_threshold(mut self, token_threshold: usize) -> Self {
        self.token_threshold = token_threshold;
        self
    }

    pub fn with_summarization_keep_recent(mut self, summarization_keep_recent: usize) -> Self {
        self.summarization_keep_recent = summarization_keep_recent;
        self
    }
}

#[async_trait::async_trait]
impl BeforeLlmCallback for ContextOptimizer {
    async fn before_llm_call(&self, session: &mut Session, llm_request: &mut LlmRequest) {
        let token_usage = count_token(&self.model, llm_request);
        if self.enable_compaction && token_usage > self.token_threshold {
            self.compact(session, llm_request);
        }
        let token_usage = count_token(&self.model, llm_request);
        if self.enbale_summarization
            && token_usage > self.token_threshold
            && let Err(err) = self.summarize(session, llm_request).await
        {
            tracing::warn!("Summarization skipped: {err}");
        }
    }
}

pub fn count_token(_model: &str, llm_request: &LlmRequest) -> usize {
    let bpe = match tiktoken_rs::o200k_base() {
        Ok(bpe) => bpe,
        Err(err) => {
            tracing::warn!("Failed to initialize tokenizer: {err}");
            return 0;
        }
    };

    let mut total = 0usize;

    for instruction in &llm_request.instructions {
        total += 4 + bpe.encode_ordinary(instruction).len();
    }

    for item in &llm_request.contents {
        total += 4;
        total += match item {
            ContentItem::Message(msg) => bpe.encode_ordinary(&msg.content).len(),
            ContentItem::ToolCall(ToolCall {
                name, arguments, ..
            }) => {
                bpe.encode_ordinary(name).len() + bpe.encode_ordinary(&arguments.to_string()).len()
            }
            ContentItem::ToolResult(tool_result) => bpe.encode_ordinary(&tool_result.content).len(),
        }
    }

    total
}
