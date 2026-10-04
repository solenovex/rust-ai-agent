use anyhow::Context;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};

use crate::{agent::context::ExecutionContext, constant::TAVILY_API_KEY_NAME, tool::Tool};

pub struct WebSearchTool;

#[async_trait::async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for current information on a given query. Default max_results is 2 (range: 1-20). Only specify max_results if you need more results."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(WebSearchArgs))
            .expect("Failed to serialize WebSearchArgs schema")
    }

    fn requires_confirmation(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: WebSearchArgs = serde_json::from_str(args_json)?;
        match search_web(args).await {
            Ok(output) => Ok(serde_json::to_string(&output)?),
            Err(err) => Ok(format!("Error: {err}")),
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct WebSearchArgs {
    pub query: String,

    #[schemars(range(min = 1, max = 20))]
    #[serde(default = "default_max_results")]
    pub max_results: u8,

    #[serde(default = "default_topic")]
    pub topic: String,

    #[serde(default)]
    pub time_range: Option<String>,

    #[serde(default)]
    pub include_raw_content: bool,
}

fn default_max_results() -> u8 {
    2
}

fn default_topic() -> String {
    "general".to_string()
}

#[derive(Debug, Serialize)]
struct TavilyRequest<'a> {
    api_key: &'a str,
    query: &'a str,
    max_results: u8,
    topic: &'a str,

    #[serde(skip_serializing_if = "Option::is_none")]
    time_range: Option<&'a str>,

    search_depth: &'a str,
    include_answer: bool,
    include_raw_content: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub content: String,
    // pub score: Option<f64>
}

#[derive(Debug, Deserialize)]
struct TavilyResponse {
    results: Vec<SearchResult>,
    answer: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct WebSearchOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,

    pub results: Vec<SearchResult>,
}

pub async fn search_web(args: WebSearchArgs) -> anyhow::Result<WebSearchOutput> {
    let api_key =
        std::env::var(TAVILY_API_KEY_NAME).context(format!("{TAVILY_API_KEY_NAME} not found"))?;

    let time_range = args
        .time_range
        .as_deref()
        .and_then(|tr| match tr.to_lowercase().as_str() {
            "day" | "week" | "month" | "year" => Some(tr),
            _ => None,
        });

    let topic = match args.topic.to_lowercase().as_str() {
        "news" => "news",
        _ => "general",
    };

    let body = TavilyRequest {
        api_key: &api_key,
        query: &args.query,
        max_results: args.max_results,
        topic,
        time_range,
        search_depth: "advanced",
        include_answer: true,
        include_raw_content: false,
    };

    let resp = reqwest::Client::new()
        .post("https://api.tavily.com/search")
        .json(&body)
        .send()
        .await
        .context("request to Tavily failed")?;

    if !resp.status().is_success() {
        anyhow::bail!("Tavily returned status {}", resp.status());
    }

    let parsed: TavilyResponse = resp
        .json()
        .await
        .context("Failed to parse Tavily response")?;

    Ok(WebSearchOutput {
        answer: parsed.answer,
        results: parsed.results,
    })
}
