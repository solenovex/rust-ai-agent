use anyhow::Ok;
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs, ResponseFormat, ResponseFormatJsonSchema,
};
use backon::{ExponentialBuilder, Retryable};
use serde::de::DeserializeOwned;
use serde_json::Value;

/// OpenAI 的 strict JSON Schema 模式对 object 节点有两条额外要求，
/// `schemars::schema_for!()` 默认都不满足，需要在发请求前手动修补：
///
/// 1. 必须显式声明 `"additionalProperties": false`。
/// 2. `"required"` 必须包含 `properties` 里的每一个键——strict 模式
///    没有“可选属性”的概念；`Option<T>` 字段本该用
///    `"type": ["T", "null"]`（nullable）来表达“可能为空”，而不是
///    把它从 required 里去掉。schemars 默认走的是后一种写法，
///    所以这里连带把 required 也强制补全。
fn enforce_strict_json_schema(schema: &mut Value) {
    match schema {
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("object") {
                map.insert("additionalProperties".to_string(), Value::Bool(false));

                if let Some(Value::Object(properties)) = map.get("properties") {
                    let all_keys: Vec<Value> = properties
                        .keys()
                        .map(|k| Value::String(k.clone()))
                        .collect();
                    map.insert("required".to_string(), Value::Array(all_keys));
                }
            }
            for value in map.values_mut() {
                enforce_strict_json_schema(value);
            }
        }
        Value::Array(items) => {
            for item in items {
                enforce_strict_json_schema(item);
            }
        }
        _ => {}
    }
}

pub async fn ask_text(model: &str, prompt: &str) -> anyhow::Result<String> {
    let client = async_openai::Client::new();
    let messages: Vec<ChatCompletionRequestMessage> = vec![
        ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()?
            .into(),
    ];

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(messages)
        .build()?;

    let response = (|| async { client.chat().create(request.clone()).await })
        .retry(ExponentialBuilder::default().with_max_times(3))
        .await?;

    response
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .ok_or_else(|| anyhow::anyhow!("No content in response"))
}

pub async fn ask_structured<T>(model: &str, prompt: &str) -> anyhow::Result<T>
where
    T: schemars::JsonSchema + DeserializeOwned,
{
    let client = async_openai::Client::new();
    let schema = schemars::schema_for!(T);
    let mut schema_json = serde_json::to_value(&schema)?;
    enforce_strict_json_schema(&mut schema_json);

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
                name: "structured_output".to_string(),
                description: None,
                schema: schema_json,
                strict: Some(true),
            },
        })
        .build()?;

    let response = (|| async { client.chat().create(request.clone()).await })
        .retry(ExponentialBuilder::default().with_max_times(3))
        .await?;

    let content = response
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .ok_or_else(|| anyhow::anyhow!("No content in response"))?;

    Ok(serde_json::from_str(&content)?)
}
