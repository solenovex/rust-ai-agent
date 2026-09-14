use std::{collections::HashMap, sync::Mutex};

use crate::session::model::Session;

#[async_trait::async_trait]
pub trait SessionManager: Send + Sync {
    async fn create(&self, session_id: &str, user_id: Option<&str>) -> anyhow::Result<Session>;
    async fn get(&self, session_id: &str) -> anyhow::Result<Option<Session>>;
    async fn save(&self, session: Session) -> anyhow::Result<()>;
    async fn get_or_create(
        &self,
        session_id: &str,
        user_id: Option<&str>,
    ) -> anyhow::Result<Session>;
}

pub struct InMemorySessionManager {
    pub sessions: Mutex<HashMap<String, Session>>,
}

impl InMemorySessionManager {
    pub fn new() -> Self {
        InMemorySessionManager {
            sessions: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for InMemorySessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl SessionManager for InMemorySessionManager {
    async fn create(&self, session_id: &str, user_id: Option<&str>) -> anyhow::Result<Session> {
        let mut guard = self.sessions.lock().unwrap();
        match guard.entry(session_id.to_owned()) {
            std::collections::hash_map::Entry::Occupied(_) => {
                anyhow::bail!("session already exists: {session_id}")
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let session = Session::new(session_id.to_owned(), user_id.map(str::to_owned));
                entry.insert(session.clone());
                Ok(session)
            }
        }
    }

    async fn get(&self, session_id: &str) -> anyhow::Result<Option<Session>> {
        Ok(self.sessions.lock().unwrap().get(session_id).cloned())
    }

    async fn save(&self, session: Session) -> anyhow::Result<()> {
        let mut guard = self.sessions.lock().unwrap();
        guard.insert(session.session_id.clone(), session);
        Ok(())
    }

    async fn get_or_create(
        &self,
        session_id: &str,
        user_id: Option<&str>,
    ) -> anyhow::Result<Session> {
        let mut guard = self.sessions.lock().unwrap();
        let session = guard
            .entry(session_id.to_owned())
            .or_insert_with(|| Session::new(session_id.to_owned(), user_id.map(str::to_string)))
            .clone();
        Ok(session)
    }
}
