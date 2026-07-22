use crate::tools::ToolBox;
use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTools, CreateChatCompletionRequestArgs,
};

pub async fn chat_complete(
    model: &str,
    system: Option<&str>,
    prompt: &str,
    toolbox: &ToolBox,
) -> anyhow::Result<String> {
    let client = async_openai::Client::new();
    let mut messages = vec![];

    if let Some(system) = system {
        messages.push(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system)
                .build()?
                .into(),
        );
    }

    messages.push(
        ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()?
            .into(),
    );

    let tool_definitions: Vec<ChatCompletionTools> = toolbox
        .values()
        .filter_map(|t| match t.definition() {
            Ok(def) => Some(def),
            Err(e) => {
                tracing::warn!("Skip tool {}, failed to get its definition: {e}", t.name());
                None
            }
        })
        .collect();

    loop {
        let request = CreateChatCompletionRequestArgs::default()
            .model(model)
            .messages(messages.clone())
            .tools(tool_definitions.clone())
            .max_tokens(2048u32)
            .build()?;

        let response = client.chat().create(request).await?;

        tracing::info!("Response: {:#?}", response);

        let message = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No choices in response"))?
            .message;

        if let Some(tool_calls) = message.tool_calls {
            messages.push(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .tool_calls(tool_calls.clone())
                    .build()?
                    .into(),
            );

            for tool_call in tool_calls {
                if let ChatCompletionMessageToolCalls::Function(function_call) = tool_call {
                    let function_name = &function_call.function.name;
                    let arguments = &function_call.function.arguments;

                    tracing::info!("Tool call: {function_name}({arguments})");

                    let tool_result = match toolbox.get(function_name) {
                        Some(tool) => match tool.execute(arguments).await {
                            Ok(result) => {
                                tracing::info!("Tool result: {result}");
                                result
                            }
                            Err(err) => {
                                let msg = format!("Tool execution error: {err}");
                                tracing::error!("{msg}");
                                msg
                            }
                        },
                        None => {
                            let msg = format!("Tool execution error: {function_name}");
                            tracing::error!("{msg}");
                            msg
                        }
                    };

                    messages.push(
                        ChatCompletionRequestToolMessageArgs::default()
                            .tool_call_id(function_call.id.clone())
                            .content(tool_result)
                            .build()?
                            .into(),
                    );
                }
            }
        } else {
            let content = message
                .content
                .ok_or_else(|| anyhow::anyhow!("No content in final response"))?;
            return Ok(content);
        }
    }
}
