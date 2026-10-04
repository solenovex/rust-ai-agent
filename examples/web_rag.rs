use study_agent::{
    search::{chunking::fixed_length_chunking, vector_search::vector_search},
    tool::web_search::{SearchResult, WebSearchArgs, search_web},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    tracing_subscriber::fmt()
        .with_target(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    let search_results = tavily_search().await?;
    let search_result_text = search_results
        .iter()
        .map(|r| format!("Title: {}\nContent: {}", r.title, r.content))
        .collect::<Vec<String>>()
        .join("\n\n");
    let bpe = tiktoken_rs::o200k_base()?;
    let tokens = bpe.encode_with_special_tokens(&search_result_text);
    println!("Total text: {}", search_result_text.len());
    println!("Total tokens: {}", tokens.len());

    let mut all_chunks: Vec<WebSearchChunk> = Vec::new();
    search_results.into_iter().for_each(|r| {
        let text = format!("Title: {}\nContent: {}", r.title, r.content);
        let chunks = fixed_length_chunking(&text, 500, 50);
        chunks.into_iter().for_each(|c| {
            all_chunks.push(WebSearchChunk {
                text: c,
                title: r.title.clone(),
                url: r.url.clone(),
            })
        });
    });

    println!("Total chunks: {}", all_chunks.len());
    let all_chunks_texts = all_chunks
        .iter()
        .map(|c| c.text.clone())
        .collect::<Vec<String>>();

    let query = "quantum computing";
    let results = vector_search(query, &all_chunks_texts, 3).await?;
    println!("Query: {query}");
    println!("{}", "=".repeat(60));
    for (i, result) in results.iter().enumerate() {
        println!("Similarity: {}, {:.3}", i + 1, result.similarity);
        print!("{}", &result.text.chars().take(300).collect::<String>());
        println!("{}", "=".repeat(60));
    }

    let selected_text = results
        .iter()
        .map(|r| r.text.to_owned())
        .collect::<Vec<String>>()
        .join("\n\n");
    let selected_tokens = bpe.encode_with_special_tokens(&selected_text).len();

    println!("Total tokens: {}", tokens.len());
    println!("Selected tokens: {}", selected_tokens);
    println!(
        "Saving rate: {:.1}%",
        (1.0 - selected_tokens as f64 / tokens.len() as f64) * 100.0
    );

    Ok(())
}

async fn tavily_search() -> anyhow::Result<Vec<SearchResult>> {
    let web_search_output = search_web(WebSearchArgs {
        query: "2025 Nobel Prize winners".to_owned(),
        max_results: 5u8,
        topic: "general".to_owned(),
        time_range: Some("year".to_owned()),
        include_raw_content: true,
    })
    .await?;

    Ok(web_search_output.results)
}

#[allow(unused)]
struct WebSearchChunk {
    pub text: String,
    pub title: String,
    pub url: String,
}
