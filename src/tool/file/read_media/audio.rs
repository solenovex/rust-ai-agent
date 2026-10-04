use std::path::Path;

use async_openai::types::chat::{
    ChatCompletionRequestMessageContentPartAudioArgs,
    ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequestArgs, InputAudio,
    InputAudioFormat,
};
use base64::Engine;

pub async fn analyze_audio(file_path: &str, query: &str, model: &str) -> anyhow::Result<String> {
    let encoded = base64::engine::general_purpose::STANDARD.encode(std::fs::read(file_path)?);
    let ext = Path::new(file_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();

    let format = match ext.as_str() {
        "wav" => InputAudioFormat::Wav,
        "mp3" => InputAudioFormat::Mp3,
        _ => anyhow::bail!("Unsupported audio format: .{ext}"),
    };

    let message = ChatCompletionRequestUserMessageArgs::default()
        .content(vec![
            ChatCompletionRequestUserMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartTextArgs::default()
                    .text(query)
                    .build()?,
            ),
            ChatCompletionRequestUserMessageContentPart::InputAudio(
                ChatCompletionRequestMessageContentPartAudioArgs::default()
                    .input_audio(InputAudio {
                        data: encoded,
                        format,
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
        .ok_or_else(|| anyhow::anyhow!("No content in audio response"))
}
