use std::sync::Arc;

use crate::{
    agent::context::ExecutionContext,
    tool::{Tool, mcp::client::McpClient},
};

pub struct McpTool {
    client: Arc<McpClient>,
    name: String,
    description: String,
    parameters: serde_json::Value,
}

impl McpTool {
    pub fn new(client: Arc<McpClient>, tool: rmcp::model::Tool) -> Self {
        let parameters = serde_json::Value::Object((*tool.input_schema).clone());
        let name = tool.name.to_string();

        Self {
            client,
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

    fn parameters(&self) -> serde_json::Value {
        self.parameters.clone()
    }

    fn requires_confirmation(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: serde_json::Value = serde_json::from_str(args_json)?;
        self.client.call_tool(&self.name, args).await
    }
}
