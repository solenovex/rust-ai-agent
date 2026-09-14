use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::agent::Event;

#[derive(Clone, Debug)]
pub struct Session {
    pub session_id: String,
    pub user_id: Option<String>,
    pub events: Vec<Event>,
    pub state: State,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
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
        }
    }
}

pub type State = HashMap<String, Value>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_session_uses_one_creation_timestamp() {
        let session = Session::new("session-1".to_string(), None);

        assert_eq!(session.created_at, session.updated_at);
    }
}
