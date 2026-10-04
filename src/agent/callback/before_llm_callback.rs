use crate::{llm::llm_request::LlmRequest, session::model::Session};

pub mod context_optimizer;
pub mod memory_injection;

#[async_trait::async_trait]
pub trait BeforeLlmCallback: Send + Sync {
    async fn before_llm_call(&self, session: &mut Session, llm_request: &mut LlmRequest);
}
