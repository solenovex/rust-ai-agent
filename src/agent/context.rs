use uuid::Uuid;

use crate::session::model::{Session, State};

use super::event::Event;

#[derive(Debug, Clone, Copy, Default)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl TokenUsage {
    pub fn add(&mut self, prompt_tokens: u32, completion_tokens: u32, total_tokens: u32) {
        self.prompt_tokens += prompt_tokens;
        self.completion_tokens += completion_tokens;
        self.total_tokens += total_tokens;
    }
}

#[derive(Debug)]
pub struct ExecutionContext {
    pub execution_id: String,
    pub current_step: u32,
    pub final_result: Option<String>,
    pub usage: TokenUsage,
    pub session: Session,
}

impl ExecutionContext {
    pub fn new(session: Session) -> Self {
        Self {
            execution_id: Uuid::new_v4().to_string(),
            current_step: 0,
            final_result: None,
            usage: TokenUsage::default(),
            session
        }
    }

    pub fn add_event(&mut self, event: Event) {
        self.session.events.push(event);
    }

    pub fn events(&self) -> Vec<Event> {
        self.session.events.clone()
    }

    pub fn state_mut(&mut self) -> &mut State {
        &mut self.session.state
    }

    pub fn increment_step(&mut self) {
        self.current_step += 1;
    }
}
