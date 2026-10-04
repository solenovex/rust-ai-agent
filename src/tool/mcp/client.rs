use rmcp::{
    RoleClient, ServiceExt,
    model::{CallToolRequestParams, Tool},
    service::RunningService,
    transport::{ConfigureCommandExt, TokioChildProcess},
};
use tokio::process::Command;

pub struct McpClient {
    service: RunningService<RoleClient, ()>,
}

impl McpClient {
    pub async fn connect() -> anyhow::Result<Self> {
        let service = ()
            .serve(TokioChildProcess::new(Command::new("cargo").configure(
                |cmd| {
                    cmd.args(["run", "--quiet", "--bin", "wikipedia_mcp_server"]);
                },
            ))?)
            .await?;
        Ok(Self { service })
    }

    pub async fn list_tools(&self) -> anyhow::Result<Vec<Tool>> {
        let result = self.service.list_tools(Default::default()).await?;
        Ok(result.tools)
    }

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> anyhow::Result<String> {
        let params = CallToolRequestParams::new(name.to_string())
            .with_arguments(arguments.as_object().cloned().unwrap_or_default());
        let result = self.service.call_tool(params).await?;

        let text = result
            .content
            .iter()
            .filter_map(|block| block.as_text().map(|t| t.text.clone()))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(text)
    }
}
