use uuid::Uuid;

use crate::session::model::Session;

#[derive(Debug, Default, Clone)]
pub struct ExecutionContext {
    pub execution_id: String,
    pub current_step: usize,
    pub final_result: Option<String>,
    pub session: Session,

    /// 待执行的转移目标。
    ///
    /// `transfer_to_agent` 工具只负责写入这个字段，真正的控制权交接由
    /// [`crate::agent::model::Agent::run_in_context`] 在一步结束后完成。
    pub transfer_to: Option<String>,

    /// 已经发生的转移次数，用来防止 agent 之间无限来回转移。
    pub transfer_count: usize,

    /// 转移路径，仅用于日志和调试，例如 `router -> math`。
    pub transfer_history: Vec<String>,
}

impl ExecutionContext {
    pub fn new(session: Session) -> Self {
        Self {
            execution_id: Uuid::new_v4().to_string(),
            current_step: 0,
            final_result: None,
            session,
            transfer_to: None,
            transfer_count: 0,
            transfer_history: Vec::new(),
        }
    }

    pub fn increment_step(&mut self) {
        self.current_step += 1;
    }
}
