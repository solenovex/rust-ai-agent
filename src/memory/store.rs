use std::sync::Mutex;

use rusqlite::Connection;

use crate::{knowledge_base::search::cosine_similarity, memory::task_memory::TaskMemory};

/// 用 Mutex 包裹 Connection：rusqlite::Connection 内部用 RefCell
/// 做单线程内部可变性，天生不满足 Sync。Tool trait 要求
/// `Send + Sync`（因为要塞进 Box<dyn Tool> 在多线程 runtime 里跑），
/// Mutex<T> 只要 T: Send 就自动是 Sync，Connection 本身是 Send 的，
/// 包一层就能满足这个约束。
pub struct TaskMemoryStore {
    conn: Mutex<Connection>,
}

impl TaskMemoryStore {
    pub fn open(path: &str) -> anyhow::Result<Self> {
        let conn = Connection::open(path)?;
        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn init_schema(conn: &Connection) -> anyhow::Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS task_memories (
                id TEXT PRIMARY KEY,
                task_summary TEXT NOT NULL,
                approach TEXT NOT NULL,
                final_answer TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                error_analysis TEXT,
                embedding_text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );",
        )?;
        Ok(())
    }

    pub fn add(&self, id: &str, memory: &TaskMemory, embedding: &[f32]) -> anyhow::Result<()> {
        let embedding_json = serde_json::to_string(embedding)?;
        let conn = self
            .conn
            .lock()
            .map_err(|_| anyhow::anyhow!("task memory store mutex poisoned"))?;
        conn.execute("INSERT INTO task_memories
             (id, task_summary, approach, final_answer, is_correct, error_analysis, embedding_text, embedding, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
              rusqlite::params![
                id,
                memory.task_summary,
                memory.approach,
                memory.final_answer,
                memory.is_correct as i64,
                memory.error_analysis,
                memory.to_embedding_text(),
                embedding_json,
                chrono::Utc::now().timestamp(),
            ],)?;

        Ok(())
    }

    pub fn query(&self, query_embedding: &[f32], top_k: usize) -> anyhow::Result<Vec<TaskMemory>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let conn = self
            .conn
            .lock()
            .map_err(|_| anyhow::anyhow!("task memory store mutex poisoned"))?;

        let mut stmt = conn.prepare(
            "SELECT task_summary, approach, final_answer, is_correct, error_analysis, embedding
             FROM task_memories",
        )?;

        let raw_rows: Vec<(TaskMemory, String)> = stmt
            .query_map([], |row| {
                let is_correct: i64 = row.get(3)?;
                Ok((
                    TaskMemory {
                        task_summary: row.get(0)?,
                        approach: row.get(1)?,
                        final_answer: row.get(2)?,
                        is_correct: is_correct != 0,
                        error_analysis: row.get(4)?,
                    },
                    row.get::<_, String>(5)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let rows: Vec<(TaskMemory, Vec<f32>)> = raw_rows
            .into_iter()
            .filter_map(|(mem, embedding_json)| {
                let embedding: Vec<f32> = serde_json::from_str(&embedding_json).ok()?;
                Some((mem, embedding))
            })
            .collect();

        let mut scored: Vec<(f32, TaskMemory)> = rows
            .into_iter()
            .map(|(mem, emb)| (cosine_similarity(query_embedding, &emb), mem))
            .collect();

        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(top_k);

        Ok(scored.into_iter().map(|(_, mem)| mem).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_memory(summary: &str) -> TaskMemory {
        TaskMemory {
            task_summary: summary.to_string(),
            approach: "used calculator".to_string(),
            final_answer: "42".to_string(),
            is_correct: true,
            error_analysis: None,
        }
    }

    #[test]
    fn add_and_query_roundtrip() -> anyhow::Result<()> {
        let store = TaskMemoryStore::open(":memory:")?;

        store.add("mem-1", &sample_memory("圆的面积怎么算"), &[1.0, 0.0, 0.0])?;
        store.add("mem-2", &sample_memory("苹果有多少种"), &[0.0, 1.0, 0.0])?;

        let results = store.query(&[0.9, 0.1, 0.0], 1)?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].task_summary, "圆的面积怎么算");
        Ok(())
    }

    #[test]
    fn query_on_empty_store_returns_empty() -> anyhow::Result<()> {
        let store = TaskMemoryStore::open(":memory:")?;
        let results = store.query(&[1.0, 0.0], 5)?;
        assert!(results.is_empty());
        Ok(())
    }
}