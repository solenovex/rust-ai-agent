use anyhow::Ok;
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

#[cfg(test)]
mod tests {
    use crate::{
        constant::TEXT_EMBEDDING_3_SMALL_MODEL,
        search::{embedding::embed_texts, similarity::cosine_similarity},
    };

    #[tokio::test]
    async fn test_embeddings() -> anyhow::Result<()> {
        dotenv::dotenv().ok();
        let sentences: Vec<String> = vec![
            "The cat is sleeping on the couch",
            "A kitten is playing with a toy",
            "The dog is running in the park",
        ]
        .into_iter()
        .map(|s| s.to_owned())
        .collect();
        let embeddings = embed_texts(&sentences, TEXT_EMBEDDING_3_SMALL_MODEL).await?;

        let cat_kitten = cosine_similarity(&embeddings[0], &embeddings[1]);
        let cat_dog = cosine_similarity(&embeddings[0], &embeddings[2]);

        println!("Cat vs Kitten: {:.3}", cat_kitten);
        println!("Cat vs Dog: {:.3}", cat_dog);

        assert!(cat_kitten > cat_dog);

        Ok(())
    }
}
