use async_trait::async_trait;
use serde_json::{Value, json};

use crate::sandbox::bridge::SandboxTool;

fn default_limit() -> u32 {
    10
}
fn default_language() -> String {
    "en".to_string()
}
fn default_output() -> String {
    "full".to_string()
}

#[derive(Debug, serde::Deserialize)]
pub struct WikiArgs {
    title: String,
    before_date: String,
    #[serde(default = "default_limit")]
    limit: u32,
    #[serde(default = "default_language")]
    language: String,
    #[serde(default = "default_output")]
    output: String,
}

/// The language is put into the host name, so only plain codes like `en` or `zh` are allowed.
pub fn validate_language(lang: &str) -> Result<(), String> {
    let ok = !lang.is_empty()
        && lang.len() <= 12
        && lang.chars().all(|c| c.is_ascii_lowercase() || c == '-');
    if ok {
        Ok(())
    } else {
        Err(format!("invalid language code: {lang:?}"))
    }
}

/// Query parameters for the revisions API: the newest revision before `before_date`.
pub fn query_params(a: &WikiArgs) -> Vec<(&'static str, String)> {
    vec![
        ("action", "query".to_string()),
        ("titles", a.title.clone()),
        ("prop", "revisions".to_string()),
        ("rvprop", "ids|timestamp|user|content".to_string()),
        ("rvslots", "main".to_string()),
        ("rvlimit", a.limit.to_string()),
        ("rvstart", a.before_date.clone()),
        ("rvdir", "older".to_string()),
        ("format", "json".to_string()),
        ("formatversion", "2".to_string()),
    ]
}

/// Fetches Wikipedia page revisions on the host, so the sandbox can stay offline.
pub struct WikipediaRevisions {
    client: reqwest::Client,
}

impl WikipediaRevisions {
    pub fn new() -> Self {
        // The Wikipedia API asks clients to send an identifying User-Agent.
        let client = reqwest::Client::builder()
            .user_agent("study-agent/0.1 (wikipedia revisions sandbox tool)")
            .build()
            .expect("reqwest client");
        Self { client }
    }
}

impl Default for WikipediaRevisions {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SandboxTool for WikipediaRevisions {
    fn name(&self) -> &str {
        "get_wikipedia_revisions"
    }

    fn definition(&self) -> Value {
        json!({
            "name": "get_wikipedia_revisions",
            "description": "Fetch revisions of a Wikipedia page from before a given ISO-8601 timestamp. Returns a list of revision objects, or a list of page texts when output='content'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "before_date": {"type": "string", "description": "e.g. 2023-07-01T00:00:00Z"},
                    "limit": {"type": "integer", "default": 10},
                    "language": {"type": "string", "default": "en"},
                    "output": {"type": "string", "enum": ["full", "content"], "default": "full"}
                },
                "required": ["title", "before_date"]
            }
        })
    }

    async fn call(&self, args: Value) -> Result<Value, String> {
        let a: WikiArgs =
            serde_json::from_value(args).map_err(|e| format!("bad arguments: {e}"))?;
        validate_language(&a.language)?;
        let url = format!("https://{}.wikipedia.org/w/api.php", a.language);
        let resp: Value = self
            .client
            .get(url)
            .query(&query_params(&a))
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json()
            .await
            .map_err(|e| e.to_string())?;

        // A missing page has no `revisions` field, which becomes an empty list.
        let revisions = resp["query"]["pages"][0]["revisions"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if a.output == "content" {
            Ok(Value::Array(
                revisions
                    .iter()
                    .map(|r| r["slots"]["main"]["content"].clone())
                    .collect(),
            ))
        } else {
            Ok(Value::Array(revisions))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::bridge::SandboxTool;

    fn args(v: Value) -> WikiArgs {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn defaults_are_applied() {
        let a = args(json!({"title": "August 1", "before_date": "2023-07-01T00:00:00Z"}));
        assert_eq!(a.limit, 10);
        assert_eq!(a.language, "en");
        assert_eq!(a.output, "full");
    }

    #[test]
    fn query_asks_for_the_last_revision_before_the_date() {
        let a =
            args(json!({"title": "August 1", "before_date": "2023-07-01T00:00:00Z", "limit": 1}));
        let p = query_params(&a);
        let get = |k: &str| p.iter().find(|(kk, _)| *kk == k).map(|(_, v)| v.clone());
        assert_eq!(get("titles").as_deref(), Some("August 1"));
        assert_eq!(get("rvstart").as_deref(), Some("2023-07-01T00:00:00Z"));
        assert_eq!(get("rvdir").as_deref(), Some("older"));
        assert_eq!(get("rvlimit").as_deref(), Some("1"));
    }

    #[test]
    fn language_must_be_a_plain_subdomain() {
        assert!(validate_language("en").is_ok());
        assert!(validate_language("zh").is_ok());
        assert!(validate_language("en.evil.com/x").is_err());
        assert!(validate_language("").is_err());
    }

    #[test]
    fn definition_names_the_tool_and_required_fields() {
        let tool = WikipediaRevisions::new();
        let def = tool.definition();
        assert_eq!(def["name"], "get_wikipedia_revisions");
        let required = def["parameters"]["required"].as_array().unwrap();
        assert!(required.contains(&json!("title")));
        assert!(required.contains(&json!("before_date")));
    }

    #[tokio::test]
    async fn bad_arguments_are_reported_without_network() {
        let tool = WikipediaRevisions::new();
        let err = tool.call(json!({"title": 5})).await.unwrap_err();
        assert!(err.starts_with("bad arguments"), "{err}");
    }

    #[tokio::test]
    async fn bad_language_is_rejected_before_any_request() {
        let tool = WikipediaRevisions::new();
        let err = tool
            .call(json!({"title": "X", "before_date": "2023-07-01T00:00:00Z", "language": "en.evil.com/x"}))
            .await
            .unwrap_err();
        assert!(err.contains("invalid language"), "{err}");
    }

    #[tokio::test]
    #[ignore = "needs network access to wikipedia.org"]
    async fn fetches_a_real_revision() {
        let tool = WikipediaRevisions::new();
        let out = tool
            .call(json!({"title": "Rust (programming language)", "before_date": "2023-07-01T00:00:00Z", "limit": 1}))
            .await
            .unwrap();
        assert_eq!(out.as_array().map(|a| a.len()), Some(1));
    }
}
