//! 在宿主机上查询 GitHub 仓库的精确 Star 数。
//!
//! 这个工具不经过沙箱，因此不受 `--network=none` 影响。让研究员用它取数，
//! 编码代理仍然可以在禁网沙箱中运行脚本。

use std::time::Duration;

use anyhow::Context;
use chrono::Utc;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{agent::context::ExecutionContext, tool::Tool};

const GITHUB_TOKEN_ENV: &str = "GITHUB_TOKEN";
const API_BASE: &str = "https://api.github.com/repos";

pub struct GithubStarsTool;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GithubStarsArgs {
    /// GitHub 仓库列表，格式为 "owner/repo"，例如 ["langchain-ai/langgraph"]。
    pub repos: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RepoStars {
    repo: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    full_name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    stars: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    archived: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GithubRepo {
    full_name: String,
    stargazers_count: u64,
    archived: bool,
    html_url: String,
}

#[async_trait::async_trait]
impl Tool for GithubStarsTool {
    fn name(&self) -> &str {
        "github_stars"
    }

    fn description(&self) -> &str {
        "Get exact current GitHub star counts for one or more repositories. \
         Use this for star numbers instead of web search snippets, rankings, \
         or old discussion posts. Accepts repositories in \"owner/repo\" form."
    }

    fn parameters(&self) -> Value {
        serde_json::to_value(schema_for!(GithubStarsArgs))
            .expect("failed to serialize GithubStarsArgs schema")
    }

    fn requires_confirmation(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: GithubStarsArgs = serde_json::from_str(args_json)?;
        if args.repos.is_empty() {
            anyhow::bail!("repos must contain at least one repository");
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .context("failed to build GitHub HTTP client")?;
        let token = std::env::var(GITHUB_TOKEN_ENV).ok();

        let mut results = Vec::with_capacity(args.repos.len());
        for raw in args.repos {
            let result = match normalize_repo(&raw) {
                Ok(repo) => fetch_repo(&client, &repo, token.as_deref()).await,
                Err(error) => RepoStars {
                    repo: raw,
                    full_name: None,
                    stars: None,
                    archived: None,
                    url: None,
                    error: Some(error),
                },
            };
            results.push(result);
        }

        Ok(serde_json::to_string_pretty(&json!({
            "queried_at": Utc::now().to_rfc3339(),
            "repos": results,
        }))?)
    }
}

async fn fetch_repo(
    client: &reqwest::Client,
    repo: &str,
    token: Option<&str>,
) -> RepoStars {
    let url = format!("{API_BASE}/{repo}");
    let mut request = client
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .header("User-Agent", "rust-ai-agent/github-stars");

    if let Some(token) = token {
        request = request.bearer_auth(token);
    }

    let response = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            return RepoStars {
                repo: repo.to_owned(),
                full_name: None,
                stars: None,
                archived: None,
                url: None,
                error: Some(format!("request failed: {error}")),
            };
        }
    };

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        let hint = if status == reqwest::StatusCode::FORBIDDEN
            || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        {
            " (rate limited; set GITHUB_TOKEN to raise the limit)"
        } else {
            ""
        };
        return RepoStars {
            repo: repo.to_owned(),
            full_name: None,
            stars: None,
            archived: None,
            url: None,
            error: Some(format!(
                "GitHub API {status}{hint}: {}",
                truncate(&body, 300)
            )),
        };
    }

    match response.json::<GithubRepo>().await {
        Ok(data) => RepoStars {
            repo: repo.to_owned(),
            full_name: Some(data.full_name),
            stars: Some(data.stargazers_count),
            archived: Some(data.archived),
            url: Some(data.html_url),
            error: None,
        },
        Err(error) => RepoStars {
            repo: repo.to_owned(),
            full_name: None,
            stars: None,
            archived: None,
            url: None,
            error: Some(format!("invalid GitHub response: {error}")),
        },
    }
}

/// 把 "owner/repo" 或 GitHub 仓库 URL 规范化为 `owner/repo`。
fn normalize_repo(input: &str) -> Result<String, String> {
    let trimmed = input.trim().trim_end_matches('/');
    let trimmed = trimmed
        .strip_prefix("https://github.com/")
        .or_else(|| trimmed.strip_prefix("http://github.com/"))
        .or_else(|| trimmed.strip_prefix("github.com/"))
        .unwrap_or(trimmed);

    let parts: Vec<&str> = trimmed.split('/').collect();
    if parts.len() != 2 {
        return Err(format!(
            "invalid repository '{input}': expected \"owner/repo\""
        ));
    }

    let (owner, repo) = (parts[0], parts[1]);
    let valid = |part: &str| {
        !part.is_empty()
            && part != "."
            && part != ".."
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };

    if !valid(owner) || !valid(repo) {
        return Err(format!(
            "invalid repository '{input}': owner and repo may only contain letters, digits, '-', '_' and '.'"
        ));
    }

    Ok(format!("{owner}/{repo}"))
}

fn truncate(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_owner_repo_and_urls() {
        assert_eq!(
            normalize_repo("langchain-ai/langgraph").unwrap(),
            "langchain-ai/langgraph"
        );
        assert_eq!(
            normalize_repo("https://github.com/crewAIInc/crewAI/").unwrap(),
            "crewAIInc/crewAI"
        );
        assert_eq!(
            normalize_repo("github.com/microsoft/autogen").unwrap(),
            "microsoft/autogen"
        );
    }

    #[test]
    fn rejects_invalid_repositories() {
        assert!(normalize_repo("langgraph").is_err());
        assert!(normalize_repo("a/b/c").is_err());
        assert!(normalize_repo("../etc/passwd").is_err());
        assert!(normalize_repo("owner/repo?tab=readme").is_err());
    }

    #[test]
    fn parameters_describe_the_repo_list() {
        let tool = GithubStarsTool;
        let params = tool.parameters();
        assert_eq!(tool.name(), "github_stars");
        assert_eq!(params["properties"]["repos"]["type"], "array");
        assert_eq!(params["properties"]["repos"]["items"]["type"], "string");
    }
}
