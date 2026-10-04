use std::{collections::HashMap, sync::Arc};

use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObjectArgs};

use crate::{
    agent::context::ExecutionContext,
    tool::{
        calculator::CalculatorTool,
        file::{
            delete::DeleteFileTool, list::ListFileTool, read::ReadFileTool,
            read_media::ReadMediaTool, unzip::UnzipFileTool,
        },
        mcp::{client::McpClient, mcp_tool::McpTool},
        web_search::WebSearchTool,
    },
};

pub mod calculator;
pub mod file;
pub mod final_answer;
pub mod mcp;
pub mod memory_tool;
pub mod web_search;

#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> serde_json::Value;
    async fn execute(&self, args_json: &str, context: &ExecutionContext) -> anyhow::Result<String>;
    fn requires_confirmation(&self) -> bool;

    fn confirmation_message_template(&self) -> &str {
        "The agent wants to execute '{name}' with arguments: {arguments}. Do you approve?"
    }

    fn get_confirmation_message(&self, arguments: &serde_json::Value) -> String {
        self.confirmation_message_template()
            .replace("{name}", self.name())
            .replace("{arguments}", &arguments.to_string())
    }

    fn definition(&self) -> anyhow::Result<ChatCompletionTools> {
        let function = FunctionObjectArgs::default()
            .name(self.name())
            .description(self.description())
            .parameters(self.parameters())
            .build()
            .map_err(|e| {
                anyhow::anyhow!("Failed to build tool definition for {}: {e}", self.name())
            })?;

        Ok(ChatCompletionTools::Function(ChatCompletionTool {
            function,
        }))
    }
}

pub type ToolBox = HashMap<String, Box<dyn Tool>>;

pub async fn build_toolbox() -> anyhow::Result<ToolBox> {
    let mut tools: Vec<Box<dyn Tool>> = vec![
        Box::new(CalculatorTool),
        Box::new(WebSearchTool),
        Box::new(UnzipFileTool),
        Box::new(ListFileTool),
        Box::new(DeleteFileTool),
        Box::new(ReadFileTool),
        Box::new(ReadMediaTool),
    ];

    let mcp_client = Arc::new(McpClient::connect().await?);
    for tool in mcp_client.list_tools().await? {
        tools.push(Box::new(McpTool::new(mcp_client.clone(), tool)));
    }

    Ok(tools
        .into_iter()
        .map(|t| (t.name().to_owned(), t))
        .collect())
}
