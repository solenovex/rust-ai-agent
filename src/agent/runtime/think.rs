use async_openai::{
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionToolChoiceOption, CreateChatCompletionRequestArgs,
        CreateChatCompletionResponse, ToolChoiceOptions,
    },
};

use crate::{agent::model::Agent, constant::GPT_6_LUNA_PRO_MODEL, llm::llm_request::LlmRequest};

impl Agent {
    pub async fn think(
        &self,
        client: &async_openai::Client<OpenAIConfig>,
        llm_request: LlmRequest,
    ) -> anyhow::Result<CreateChatCompletionResponse> {
        let messages = llm_request.build_message()?;
        let tool_choice = match llm_request.tool_choice {
            crate::llm::llm_request::ToolChoice::Auto => {
                ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Auto)
            }
            crate::llm::llm_request::ToolChoice::Required => {
                ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Required)
            }
        };
        let request = CreateChatCompletionRequestArgs::default()
            .model(GPT_6_LUNA_PRO_MODEL)
            .max_tokens(2048u32)
            .tools(self.tool_definitions())
            .tool_choice(tool_choice)
            .messages(messages)
            .build()?;

        let response = client.chat().create(request).await?;
        Ok(response)
    }
}
