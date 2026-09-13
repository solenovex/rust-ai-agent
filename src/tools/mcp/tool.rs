use std::sync::Arc;

use serde_json::Value;

use crate::{
    agent::ExecutionContext,
    tools::{mcp::client::McpClient, tool::Tool},
};

/// 把一个 MCP 工具包装成 Agent 认识的 Tool trait。
/// 一个 McpTool 对应 MCP Server 的 list_tools() 里的一条工具信息，
/// execute 的时候再转发给 McpClient::call_tool，
/// 对 Agent loop 来说，它跟 calculator / web_search 没有任何区别。
pub struct McpTool {
    client: Arc<McpClient>,
    name: String,
    description: String,
    parameters: Value,
    requires_confirmation: bool,
}

impl McpTool {
    /// 从 rmcp 的 Tool（协议里的原始工具描述）转换成我们自己的 McpTool。
    /// name / description 转成 owned String，是因为 Tool trait 要求
    /// name(&self) -> &str 返回的引用要跟 self 的生命周期绑在一起，
    /// 不能直接借用 rmcp::model::Tool 里 'static 的 Cow<str>。
    pub fn new(client: Arc<McpClient>, tool: rmcp::model::Tool) -> Self {
        let parameters = Value::Object((*tool.input_schema).clone());
        let name = tool.name.to_string();

        Self {
            client,
            requires_confirmation: name == "delete_expense",
            name,
            description: tool.description.map(|d| d.to_string()).unwrap_or_default(),
            parameters,
        }
    }
}

#[async_trait::async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    fn requires_confirmation(&self) -> bool {
        self.requires_confirmation
    }

    fn confirmation_message_template(&self) -> &str {
        "即将调用费用工具 '{name}'，参数为 {arguments}。这是不可逆操作，是否继续？"
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        // 大模型给的参数是 JSON 字符串，先解析成 Value，
        // 再转发给 McpClient::call_tool —— 之后的事情
        // （发给 MCP Server，Server 再转发给 expense-tracker-api）
        // 跟这里的 execute 方法就没关系了
        let args: Value = serde_json::from_str(args_json)?;
        self.client.call_tool(&self.name, args).await
    }
}
