use std::{path::PathBuf, sync::OnceLock};

use async_openai::types::chat::{
    ChatCompletionRequestMessageContentPartImageArgs,
    ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequestArgs, ImageUrl,
};

use pdfium_render::prelude::*;

static PDFIUM: OnceLock<std::result::Result<Pdfium, String>> = OnceLock::new();

fn pdfium() -> anyhow::Result<&'static Pdfium> {
    let initialized = PDFIUM.get_or_init(|| {
        let library_path = std::env::var_os("PDFIUM_LIB_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Pdfium::pdfium_platform_library_name_at_path(env!("CARGO_MANIFEST_DIR"))
            });

        Pdfium::bind_to_library(&library_path)
            .map(Pdfium::new)
            .map_err(|err| format!("Unable to load Pdfium at {}: {err}", library_path.display()))
    });

    initialized
        .as_ref()
        .map_err(|err| anyhow::anyhow!(err.clone()))
}

pub async fn analyze_pdf(file_path: &str, query: &str, model: &str) -> anyhow::Result<String> {
    use base64::Engine;

    let document = pdfium()?.load_pdf_from_file(file_path, None)?;

    // Extract text from all pages.
    let mut text_content = String::new();

    for page in document.pages().iter() {
        let text = page.text()?.all();
        text_content.push_str(&text);
    }

    // Render first 5 pages as PNG and base64 encode them.
    let render_config = PdfRenderConfig::new().set_target_width(1600);

    let extracted_text: String = text_content.chars().take(3000).collect();
    let mut content = vec![ChatCompletionRequestUserMessageContentPart::Text(
        ChatCompletionRequestMessageContentPartTextArgs::default()
            .text(format!("{}\n\nExtracted text:\n{}", query, extracted_text))
            .build()?,
    )];

    for page in document.pages().iter().take(5) {
        let image = page.render_with_config(&render_config)?.as_image()?;

        let mut png_bytes = Vec::new();

        {
            use std::io::Cursor;
            image.write_to(&mut Cursor::new(&mut png_bytes), image::ImageFormat::Png)?;
        }

        let encoded = base64::engine::general_purpose::STANDARD.encode(png_bytes);

        let data_url = format!("data:image/png;base64,{encoded}");

        content.push(ChatCompletionRequestUserMessageContentPart::ImageUrl(
            ChatCompletionRequestMessageContentPartImageArgs::default()
                .image_url(ImageUrl {
                    url: data_url,
                    detail: None,
                })
                .build()?,
        ));
    }

    let message = ChatCompletionRequestUserMessageArgs::default()
        .content(content)
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
        .ok_or_else(|| anyhow::anyhow!("No content in PDF response"))
}
