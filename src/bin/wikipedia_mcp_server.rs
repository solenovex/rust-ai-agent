use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ContentBlock, ServerCapabilities, ServerConfig},
    tool, tool_handler, tool_router,
    transport::stdio,
};
use schemars::JsonSchema;
use serde::Deserialize;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    let server = WikiServer::new();
    let service = server.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}

#[derive(Debug, Deserialize)]
struct WikiSummary {
    #[serde(default)]
    titles: Option<Titles>,

    #[serde(default)]
    description: Option<String>,

    extract: String,

    #[serde(default)]
    thumbnail: Option<Thumbnail>,

    #[serde(default)]
    content_urls: Option<ContentUrls>,
}

#[derive(Debug, Deserialize)]
struct Titles {
    display: String,
}

#[derive(Debug, Deserialize)]
struct Thumbnail {
    source: String,
}

#[derive(Debug, Deserialize)]
struct ContentUrls {
    desktop: DesktopUrl,
}

#[derive(Debug, Deserialize)]
struct DesktopUrl {
    page: String,
}

#[derive(Debug, Deserialize)]
struct WikiProblem {
    #[serde(default)]
    _title: String,

    #[serde(default)]
    detail: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SummaryRequest {
    title: String,
}

struct WikiServer {
    client: reqwest::Client,
    base_url: String,

    #[allow(dead_code)]
    tool_router: ToolRouter<WikiServer>,
}

#[tool_router]
impl WikiServer {
    fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("study-agent/0.1 (contact: 331335713@qq.com)")
                .build()
                .expect("failed to build http client"),
            base_url: std::env::var("WIKIPEDIA_API_URL").unwrap_or_else(|_| {
                "https://en.wikipedia.org/api/rest_v1/page/summary/".to_string()
            }),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Get a short summary and intro extract for a Wikipedia article by title.")]
    async fn wiki_summary(
        &self,
        Parameters(req): Parameters<SummaryRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let binding = req.title.replace(' ', "_");
        let encoded = urlencoding::encode(&binding);
        let url = format!("{}{encoded}", self.base_url);

        let resp =
            self.client.get(&url).send().await.map_err(|e| {
                rmcp::ErrorData::internal_error(format!("request failed: {e}"), None)
            })?;

        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            let problem: WikiProblem = resp.json().await.unwrap_or(WikiProblem {
                _title: "Not Found".into(),
                detail: "No matching page.".into(),
            });
            return Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "No Wikipedia page found for \"{}\": {}",
                req.title, problem.detail
            ))]));
        }

        if !status.is_success() {
            return Err(rmcp::ErrorData::internal_error(
                format!("Wikipedia API returned status {status}"),
                None,
            ));
        }

        let summary: WikiSummary = resp.json().await.map_err(|e| {
            rmcp::ErrorData::internal_error(format!("failed to parse response: {e}"), None)
        })?;

        let display_title = summary
            .titles
            .map(|t| t.display)
            .unwrap_or_else(|| req.title.clone());

        let mut text = format!("# {display_title}\n\n");
        if let Some(desc) = summary.description {
            text.push_str(&format!("_{desc}_\n\n"));
        }
        text.push_str(&summary.extract);
        if let Some(urls) = summary.content_urls {
            text.push_str(&format!("\n\nSource: {}", urls.desktop.page));
        }
        let mut contents = vec![ContentBlock::text(text)];
        if let Some(thumb) = summary.thumbnail {
            contents.push(ContentBlock::text(format!("Thumbnail: {}", thumb.source)));
        }

        Ok(CallToolResult::success(contents))
    }
}

#[tool_handler]
impl ServerHandler for WikiServer {
    fn get_info(&self) -> ServerConfig {
        let mut config = ServerConfig::default();
        config.capabilities = ServerCapabilities::builder().enable_tools().build();
        config.instructions = Some("Provides Wikipedia lookup tools.".into());
        config
    }
}
