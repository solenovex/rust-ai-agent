use std::collections::{HashMap, hash_map::Entry};

use anyhow::Ok;

use crate::session::model::Session;

#[async_trait::async_trait]
pub trait SessionManager {
    async fn create(&mut self, session_id: &str, user_id: Option<&str>) -> anyhow::Result<Session>;
    async fn get(&self, session_id: &str) -> anyhow::Result<Option<Session>>;
    async fn save(&mut self, session: Session) -> anyhow::Result<()>;
    async fn get_or_create(
        &mut self,
        session_id: &str,
        user_id: Option<&str>,
    ) -> anyhow::Result<Session>;
}

pub struct InMemorySessionManager {
    pub sessions: HashMap<String, Session>,
}

#[async_trait::async_trait]
impl SessionManager for InMemorySessionManager {
    async fn create(&mut self, session_id: &str, user_id: Option<&str>) -> anyhow::Result<Session> {
        if self.sessions.contains_key(session_id) {
            anyhow::bail!("session already exists: {session_id}")
        }

        let session = Session::new(session_id.to_owned(), user_id.map(str::to_owned));
        match self.sessions.entry(session_id.to_owned()) {
            Entry::Occupied(_) => {
                anyhow::bail!("session already exists: {session_id}")
            }
            Entry::Vacant(entry) => {
                entry.insert(session.clone());
                Ok(session)
            }
        }
    }

    async fn get(&self, session_id: &str) -> anyhow::Result<Option<Session>> {
        let one: Option<Session> = self.sessions.get(session_id).cloned();
        Ok(one)
    }

    async fn save(&mut self, session: Session) -> anyhow::Result<()> {
        self.sessions.insert(session.session_id.clone(), session);
        Ok(())
    }

    async fn get_or_create(&mut self, session_id: &str, user_id: Option<&str>) -> anyhow::Result<Session> {
        if let Some(session) = self.get(session_id).await? {
            return Ok(session);
        }

        self.create(session_id, user_id).await
    }
}
