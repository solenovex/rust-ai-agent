use std::path::Path;

use async_openai::types::chat::{
    ChatCompletionRequestMessageContentPartImageArgs,
    ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequestArgs, ImageUrl,
};
use base64::Engine;

pub async fn analyze_image(file_path: &str, query: &str, model: &str) -> anyhow::Result<String> {
    let encoded = base64::engine::general_purpose::STANDARD.encode(std::fs::read(file_path)?);
    let mime = image_mime_type(file_path)?;
    let data_url = format!("data:{mime};base64,{encoded}");

    let message = ChatCompletionRequestUserMessageArgs::default()
        .content(vec![
            ChatCompletionRequestUserMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartTextArgs::default()
                    .text(query)
                    .build()?,
            ),
            ChatCompletionRequestUserMessageContentPart::ImageUrl(
                ChatCompletionRequestMessageContentPartImageArgs::default()
                    .image_url(ImageUrl {
                        url: data_url,
                        detail: None,
                    })
                    .build()?,
            ),
        ])
        .build()?;
    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(vec![message.into()])
        .max_tokens(1024u32)
        .build()?;

    let client = async_openai::Client::new();
    let response = client.chat().create(request).await?;

    response
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .ok_or_else(|| anyhow::anyhow!("No content in vision response"))
}

fn image_mime_type(file_path: &str) -> anyhow::Result<&'static str> {
    let ext = Path::new(file_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "jpg" | "jpeg" => Ok("image/jpeg"),
        "png" => Ok("image/png"),
        "gif" => Ok("image/gif"),
        "webp" => Ok("image/webp"),
        "bmp" => Ok("image/bmp"),
        "tif" | "tiff" => Ok("image/tiff"),
        _ => anyhow::bail!("Unsupported image type: .{}", ext),
    }
}
