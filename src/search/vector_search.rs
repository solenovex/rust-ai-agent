use crate::{
    constant::TEXT_EMBEDDING_3_SMALL_MODEL,
    search::{
        embedding::{embed_text, embed_texts},
        similarity::cosine_similarity,
    },
};

#[derive(Debug)]
pub struct SearchHit {
    pub text: String,
    pub similarity: f32,
}

pub async fn vector_search(
    query: &str,
    chunks: &[String],
    top_k: usize,
) -> anyhow::Result<Vec<SearchHit>> {
    dotenv::dotenv().ok();
    let query_embedding = embed_text(query, TEXT_EMBEDDING_3_SMALL_MODEL).await?;
    let chunks_embedding = embed_texts(chunks, TEXT_EMBEDDING_3_SMALL_MODEL).await?;
    let mut ranks: Vec<(usize, f32)> = chunks_embedding
        .iter()
        .enumerate()
        .map(|(i, ce)| (i, cosine_similarity(&query_embedding, ce)))
        .collect();
    ranks.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top_chunks: Vec<SearchHit> = ranks
        .into_iter()
        .map(|(i, similarity)| SearchHit {
            text: chunks[i].clone(),
            similarity,
        })
        .take(top_k)
        .collect();

    Ok(top_chunks)
}

#[cfg(test)]
mod vector_search_tests {
    use super::*;

    #[tokio::test]
    async fn test_vector_search() -> anyhow::Result<()> {
        let documents: Vec<String> = [
            "Python is a programming language",
            "Machine learning uses Python extensively",
            "Cats are popular pets",
            "Deep learning is a subset of machine learning",
        ]
        .into_iter()
        .map(|s| s.to_owned())
        .collect();

        let results = vector_search("Artificial Intelligence", &documents, 4).await?;
        assert!(results[0].text == documents[3]);
        assert!(results[1].text == documents[1]);
        assert!(results[2].text == documents[0]);
        assert!(results[3].text == documents[2]);

        Ok(())
    }
}
