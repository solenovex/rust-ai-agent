use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObjectArgs};
use serde_json::Value;

use crate::agent::ExecutionContext;

#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;

    fn description(&self) -> &str;

    fn parameters(&self) -> Value;

    async fn execute(&self, args_json: &str, context: &ExecutionContext) -> anyhow::Result<String>;

    fn requires_confirmation(&self) -> bool {
        false
    }

    fn confirmation_message_template(&self) -> &str {
        "The agent wants to execute '{name}' with arguments: {arguments}. Do you approve?"
    }

    fn get_confirmation_message(&self, arguments: &Value) -> String {
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
