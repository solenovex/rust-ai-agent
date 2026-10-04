use std::collections::HashMap;

use chrono::Utc;

use crate::agent::{event::AgentEvent, tool_confirmation::PendingToolCall};

#[derive(Clone, Debug, Default)]
pub struct Session {
    pub session_id: String,
    pub user_id: Option<String>,
    pub events: Vec<AgentEvent>,
    pub state: State,
    pub created_at: chrono::DateTime<Utc>,
    pub updated_at: chrono::DateTime<Utc>,
    pub pending_tool_calls: Vec<PendingToolCall>,
}

impl Session {
    pub fn new(session_id: String, user_id: Option<String>) -> Self {
        let now = Utc::now();
        Session {
            session_id,
            user_id,
            events: Vec::new(),
            state: HashMap::new(),
            created_at: now,
            updated_at: now,
            pending_tool_calls: Vec::new(),
        }
    }

    pub fn add_event(&mut self, event: AgentEvent) {
        self.events.push(event);
    }

    pub fn add_pending_tool_call(&mut self, pending_tool_call: PendingToolCall) {
        self.pending_tool_calls.push(pending_tool_call);
    }
}

pub type State = HashMap<String, serde_json::Value>;
