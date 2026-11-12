use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls, ChatCompletionRequestMessage,
    ChatCompletionRequestToolMessageArgs,
};

use crate::{
    agent::{
        context::ExecutionContext,
        event::{AgentEvent, ContentItem, ToolCall, ToolResult, ToolResultStatus},
        model::Agent,
        tool_confirmation::PendingToolCall,
    },
    constant::TRANSFER_TOOL_NAME,
    tool::TransferTool,
};

impl Agent {
    pub async fn act_all(
        &self,
        tool_calls: &[ChatCompletionMessageToolCalls],
        context: &mut ExecutionContext,
    ) -> anyhow::Result<()> {
        for tool_call in tool_calls {
            match tool_call {
                ChatCompletionMessageToolCalls::Function(tool_call) => {
                    // 转移工具不属于普通 toolbox，需要先拦截：
                    // 它不执行外部操作，只把目标写入 context.transfer_to。
                    if tool_call.function.name == TRANSFER_TOOL_NAME {
                        self.act_transfer(tool_call, context).await?;
                        continue;
                    }

                    if let Some(tool) = self.toolbox.get(&tool_call.function.name)
                        && tool.requires_confirmation()
                    {
                        let arguments = serde_json::from_str(&tool_call.function.arguments)
                            .unwrap_or(serde_json::Value::Null);
                        context.session.add_event(AgentEvent::new(
                            context.execution_id.clone(),
                            self.name.clone(),
                            vec![ContentItem::ToolCall(ToolCall::new(
                                tool_call.id.clone(),
                                tool_call.function.name.clone(),
                                arguments.clone(),
                            ))],
                        ));
                        context.session.add_pending_tool_call(PendingToolCall::new(
                            tool_call.clone(),
                            tool.get_confirmation_message(&arguments),
                        ));
                        continue;
                    }
                    self.act(tool_call, context).await?;
                }
                ChatCompletionMessageToolCalls::Custom(custom_tool_call) => {
                    tracing::info!("Custom tool call not implemented");
                    context.session.add_event(AgentEvent::new(
                        context.execution_id.clone(),
                        self.name.clone(),
                        vec![ContentItem::ToolResult(ToolResult::new(
                            custom_tool_call.id.clone(),
                            custom_tool_call.custom_tool.name.clone(),
                            ToolResultStatus::Error,
                            "Custom tool call not implemented".to_owned(),
                        ))],
                    ));
                }
            }
        }

        Ok(())
    }

    /// 处理 `transfer_to_agent` 调用。
    ///
    /// 只记录一次转移请求并写入上下文；真正的 `run_in_context` 交接发生在本步
    /// 结束之后。这样同一条 assistant 消息里的其它工具调用仍然能正常执行，
    /// 不会产生“有 tool_call 却没有 tool_result”的非法消息序列。
    async fn act_transfer(
        &self,
        tool_call: &ChatCompletionMessageToolCall,
        context: &mut ExecutionContext,
    ) -> anyhow::Result<()> {
        let tool_call_id = tool_call.id.clone();
        let args_json = tool_call.function.arguments.clone();
        let arguments = serde_json::from_str(&args_json).unwrap_or(serde_json::Value::Null);

        context.session.add_event(AgentEvent::new(
            context.execution_id.clone(),
            self.name.clone(),
            vec![ContentItem::ToolCall(ToolCall::new(
                tool_call_id.clone(),
                TRANSFER_TOOL_NAME.to_owned(),
                arguments,
            ))],
        ));

        tracing::info!(
            "Transfer request ({tool_call_id}) from {}: {args_json}",
            self.name
        );

        // 每次调用都根据当前 agent 树生成工具，确保 parent/sibling 关系
        // 在树组装完成后是最新的。
        let transfer_tool = TransferTool::new(self.transfer_targets());
        let (content, status) = match transfer_tool.resolve(&args_json) {
            Ok(target) => {
                // 与书中一致：一次执行中只接受第一个转移请求，避免同一步里
                // 多个工具调用互相覆盖目标。
                if let Some(existing) = &context.transfer_to {
                    (
                        format!("Transfer already requested to {existing}; ignoring this request."),
                        ToolResultStatus::Success,
                    )
                } else {
                    tracing::info!("Transfer accepted: {} -> {}", self.name, target.name);
                    context.transfer_to = Some(target.name.clone());
                    (
                        format!("Transferring to {}...", target.name),
                        ToolResultStatus::Success,
                    )
                }
            }
            Err(err) => (format!("Transfer failed: {err}"), ToolResultStatus::Error),
        };

        context.session.add_event(AgentEvent::new(
            context.execution_id.clone(),
            self.name.clone(),
            vec![ContentItem::ToolResult(ToolResult::new(
                tool_call_id,
                TRANSFER_TOOL_NAME.to_owned(),
                status,
                content,
            ))],
        ));

        Ok(())
    }

    async fn act(
        &self,
        tool_call: &ChatCompletionMessageToolCall,
        context: &mut ExecutionContext,
    ) -> anyhow::Result<ChatCompletionRequestMessage> {
        let tool_call_id = tool_call.id.clone();
        let tool_name = tool_call.function.name.clone();
        let args_json = tool_call.function.arguments.clone();

        context.session.add_event(AgentEvent::new(
            context.execution_id.clone(),
            self.name.clone(),
            vec![ContentItem::ToolCall(ToolCall::new(
                tool_call_id.clone(),
                tool_name.clone(),
                serde_json::from_str(&args_json)?,
            ))],
        ));

        tracing::info!("Tool call request ({tool_call_id}): {tool_name} with args: {args_json}");

        let tool = self.toolbox.get(&tool_name);

        let mut before_callback_result = None;
        for callback in self.before_tool_callbacks.iter() {
            before_callback_result = callback.call(context, tool_call).await;
            if before_callback_result.is_some() {
                break;
            }
        }

        if let Some(before_callback_result) = before_callback_result {
            context.session.add_event(AgentEvent::new(
                context.execution_id.clone(),
                self.name.clone(),
                vec![ContentItem::ToolResult(ToolResult::new(
                    tool_call_id.clone(),
                    tool_name.clone(),
                    ToolResultStatus::Success,
                    before_callback_result.clone(),
                ))],
            ));

            Ok(ChatCompletionRequestToolMessageArgs::default()
                .tool_call_id(tool_call_id.clone())
                .content(before_callback_result)
                .build()?
                .into())
        } else {
            let tool_result = match tool {
                Some(tool) => tool.execute(&args_json, context).await,
                None => Ok(format!(
                    "Tool error: unable to get tool: {tool_name} from ToolBox."
                )),
            };

            let (status, content) = match tool_result {
                Ok(result) => {
                    tracing::info!("Tool call result ({tool_call_id}): {result}");
                    (ToolResultStatus::Success, result)
                }
                Err(err) => {
                    let msg = format!("Tool call failed ({tool_call_id}): {}", err);
                    tracing::info!("{msg}");
                    (ToolResultStatus::Error, msg)
                }
            };

            let mut after_callback_result = None;
            for callback in self.after_tool_callbacks.iter() {
                after_callback_result = callback.call(context, tool_call, &content).await;
                if after_callback_result.is_some() {
                    break;
                }
            }

            let result = after_callback_result.unwrap_or(content);

            context.session.add_event(AgentEvent::new(
                context.execution_id.clone(),
                self.name.clone(),
                vec![ContentItem::ToolResult(ToolResult::new(
                    tool_call_id.clone(),
                    tool_name.clone(),
                    status,
                    result.clone(),
                ))],
            ));

            Ok(ChatCompletionRequestToolMessageArgs::default()
                .tool_call_id(tool_call_id.clone())
                .content(result.clone())
                .build()?
                .into())
        }
    }
}
