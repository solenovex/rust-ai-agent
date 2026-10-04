use anyhow::Context;
use async_openai::{
    error::OpenAIError,
    types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs, ResponseFormat, ResponseFormatJsonSchema,
    },
};
use backon::{ExponentialBuilder, Retryable};
use serde::de::DeserializeOwned;

pub async fn ask<T>(model: &str, prompt: &str) -> anyhow::Result<T>
where
    T: schemars::JsonSchema + DeserializeOwned,
{
    let client = async_openai::Client::new();

    let mut schema = serde_json::to_value(schemars::schema_for!(T))?;
    enforce_strict_schema(&mut schema);

    let messages: Vec<ChatCompletionRequestMessage> = vec![
        ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()?
            .into(),
    ];

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(messages)
        .response_format(ResponseFormat::JsonSchema {
            json_schema: ResponseFormatJsonSchema {
                name: "response".to_string(),
                description: None,
                schema,
                strict: Some(true),
            },
        })
        .build()?;

    let response = (|| async { client.chat().create(request.clone()).await })
        .retry(ExponentialBuilder::default().with_max_times(3))
        .when(is_retryable)
        .await
        .context("LLM request failed")?;

    let message = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("LLM returned no choices"))?
        .message;

    if let Some(refusal) = message.refusal {
        return Err(anyhow::anyhow!("LLM refused to answer: {refusal}"));
    }

    let content = message
        .content
        .ok_or_else(|| anyhow::anyhow!("LLM returned empty content"))?;

    serde_json::from_str(&content).with_context(|| format!("failed to parse LLM output: {content}"))
}

fn enforce_strict_schema(schema: &mut serde_json::Value) {
    match schema {
        serde_json::Value::Object(map) => {
            if map.get("type").and_then(serde_json::Value::as_str) == Some("object") {
                map.insert(
                    "additionalProperties".into(),
                    serde_json::Value::Bool(false),
                );
                if let Some(serde_json::Value::Object(props)) = map.get("properties") {
                    let keys: Vec<serde_json::Value> = props
                        .keys()
                        .map(|k| serde_json::Value::String(k.clone()))
                        .collect();
                    map.insert("required".into(), serde_json::Value::Array(keys));
                }
            }
            for v in map.values_mut() {
                enforce_strict_schema(v);
            }
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(enforce_strict_schema),
        _ => {}
    }
}

fn is_retryable(e: &OpenAIError) -> bool {
    match e {
        OpenAIError::Reqwest(_) => true,
        OpenAIError::ApiError(api) => matches!(
            api.api_error.r#type.as_deref(),
            Some("rate_limit_exceeded" | "server_error")
        ),
        _ => false,
    }
}
