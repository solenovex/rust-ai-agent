use crate::{
    agent::{
        ContentItem, ExecutionContext,
        llm_request::{BeforeLlmCallback, LlmRequest},
    }, callback::context_optimizer::{count_tokens, find_safe_start},
};

pub struct SlidingWindow {
    pub model: String,
    pub token_threshold: usize,
    pub window_size: usize,
}

#[async_trait::async_trait]
impl BeforeLlmCallback for SlidingWindow {
    async fn call(&self, _context: &mut ExecutionContext, request: &mut LlmRequest) {
        if count_tokens(&self.model, request) < self.token_threshold {
            return;
        }

        if request.contents.len() <= self.window_size {
            return;
        }

        let user_idx = request
            .contents
            .iter()
            .position(|item| matches!(item, ContentItem::Message { role , .. } if role == "user"));
        let Some(user_idx) = user_idx else { return };

        let start= find_safe_start(&request.contents, request.contents.len() - self.window_size);
        let start = start.max(user_idx  + 1);

        request.contents.drain(user_idx + 1 .. start);
    }
}
