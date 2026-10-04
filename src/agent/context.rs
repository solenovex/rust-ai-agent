use uuid::Uuid;

use crate::session::model::Session;

#[derive(Debug, Default)]
pub struct ExecutionContext {
    pub execution_id: String,
    pub current_step: usize,
    pub final_result: Option<String>,
    pub session: Session,
}

impl ExecutionContext {
    pub fn new(session: Session) -> Self {
        Self {
            execution_id: Uuid::new_v4().to_string(),
            current_step: 0,
            final_result: None,
            session,
        }
    }

    pub fn increment_step(&mut self) {
        self.current_step += 1;
    }
}
