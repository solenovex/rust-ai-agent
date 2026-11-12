use serde::de::DeserializeOwned;

use crate::agent::{
    context::ExecutionContext,
    event::{AgentEvent, ContentItem, Message},
    model::{Agent, AgentResult, AgentStatus},
    runtime::step::StepResult,
    tool_confirmation::ToolConfirmation,
};

pub mod act;
pub mod step;
pub mod think;

/// 一次执行中允许的最大转移次数，防止 agent 之间无限来回转移。
const MAX_TRANSFERS: usize = 8;

impl Agent {
    /// 在已有的上下文上运行循环。工作流内部会调用它。
    pub async fn run_in_context(
        &self,
        mut context: ExecutionContext,
    ) -> anyhow::Result<AgentResult<serde_json::Value>> {
        let client = async_openai::Client::new();
        loop {
            if context.current_step >= self.max_steps {
                anyhow::bail!(
                    "Agent exceeded the maximum of {} steps without a final answer",
                    self.max_steps
                );
            }

            match self.step(&client, &mut context).await? {
                StepResult::Continue => {
                    context.increment_step();

                    // 审批优先于转移：如果同一步里既请求了危险工具又请求了
                    // transfer，当前 agent 必须先处理审批，否则 pending 会
                    // 留在错误的 agent 上。
                    if !context.session.pending_tool_calls.is_empty() {
                        self.session_manager.save(context.session.clone()).await?;
                        return Ok(AgentResult::new(
                            None,
                            context,
                            AgentStatus::PendingConfirmation,
                        ));
                    }

                    // 一步结束后检查转移请求。此时 transfer_to_agent 的
                    // tool_call/tool_result 已经写入 conversation，接手的
                    // agent 会在下一次 LLM 请求中看到完整的交接过程。
                    if let Some(target_name) = context.transfer_to.take() {
                        if context.transfer_count >= MAX_TRANSFERS {
                            anyhow::bail!(
                                "agent transfer exceeded the maximum of {MAX_TRANSFERS} handoffs"
                            );
                        }

                        let next_agent =
                            self.find_transfer_target(&target_name).ok_or_else(|| {
                                anyhow::anyhow!(
                                    "agent '{}' cannot transfer to '{}'; allowed targets: {:?}",
                                    self.name,
                                    target_name,
                                    self.transfer_targets()
                                        .iter()
                                        .map(|agent| agent.name.clone())
                                        .collect::<Vec<_>>()
                                )
                            })?;

                        // 每个 agent 有自己的步数预算；会话/事件则完全共享。
                        context.current_step = 0;
                        context.transfer_count += 1;
                        context
                            .transfer_history
                            .push(format!("{} -> {}", self.name, next_agent.name));

                        tracing::info!(
                            "Transferring control: {} -> {} (handoff #{})",
                            self.name,
                            next_agent.name,
                            context.transfer_count
                        );

                        // 交接前保存一次，避免接手的 agent 中途失败时丢失会话。
                        self.session_manager.save(context.session.clone()).await?;

                        // 递归调用会产生无限大小的 Future，必须装箱。
                        // 返回下一个 agent 的结果，意味着控制权已经彻底交给它，
                        // 原 agent 不再回复用户。
                        return Box::pin(next_agent.run_in_context(context)).await;
                    }
                }
                StepResult::Final(final_result) => {
                    let value = if self.has_final_answer_tool() {
                        serde_json::from_str::<serde_json::Value>(&final_result)?
                    } else {
                        serde_json::Value::String(final_result)
                    };
                    self.session_manager.save(context.session.clone()).await?;
                    return Ok(AgentResult::new(
                        Some(value),
                        context,
                        AgentStatus::Complete,
                    ));
                }
            }
        }
    }

    /// 原来的 run 现在是薄包装：创建上下文 → 加入用户消息 → 调用 run_in_context。
    pub async fn run<T: DeserializeOwned>(
        &self,
        user_prompt: &str,
        session_id: &str,
        tool_confirmations: Option<Vec<ToolConfirmation>>,
    ) -> anyhow::Result<AgentResult<T>> {
        let session = self.session_manager.get_or_create(session_id, None).await?;
        let mut context = ExecutionContext::new(session);

        push_user_message(&mut context, user_prompt);

        if let Some(confirmations) = tool_confirmations {
            let value = serde_json::to_value(&confirmations)?;
            context
                .session
                .state
                .insert("tool_confirmations".to_string(), value);
        }

        println!("You> {user_prompt}");

        let result = self.run_in_context(context).await?;

        let output = match result.output {
            Some(value) => serde_json::from_value::<Option<T>>(value)?,
            None => None,
        };

        Ok(AgentResult::new(output, result.context, result.status))
    }
}

/// 把用户消息加入上下文。空字符串会被忽略。
pub fn push_user_message(context: &mut ExecutionContext, text: &str) {
    if !text.is_empty() {
        let user_event = AgentEvent::new(
            context.execution_id.clone(),
            "user",
            vec![ContentItem::Message(Message::new(
                "user".to_owned(),
                text.to_owned(),
            ))],
        );
        context.session.add_event(user_event);
    }
}
