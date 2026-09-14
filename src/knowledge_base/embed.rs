use async_openai::types::embeddings::{CreateEmbeddingRequestArgs, EmbeddingInput};

pub async fn embed_texts(texts: &[String], model: &str) -> anyhow::Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let client = async_openai::Client::new();
    let request = CreateEmbeddingRequestArgs::default()
        .model(model)
        .input(EmbeddingInput::StringArray(texts.to_vec()))
        .build()?;

    let response = client.embeddings().create(request).await?;

    let mut data = response.data;
    data.sort_by_key(|embedding| embedding.index);

    Ok(data
        .into_iter()
        .map(|embedding| embedding.embedding)
        .collect())
}

pub async fn embed_text(text: &str, model: &str) -> anyhow::Result<Vec<f32>> {
    let owned = [text.to_string()];
    let mut vectors = embed_texts(&owned, model).await?;
    vectors
        .pop()
        .ok_or_else(|| anyhow::anyhow!("embedding API returned no vectors"))
}
