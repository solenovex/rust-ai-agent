use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};

use crate::{agent::context::ExecutionContext, tool::Tool};

pub struct AlwaysFailingWikipediaTool;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct WikipediaArgs {
    pub title: String,
}

#[async_trait::async_trait]
impl Tool for AlwaysFailingWikipediaTool {
    fn name(&self) -> &str {
        "get_wikipedia_page"
    }

    fn description(&self) -> &str {
        "Get a Wikipedia page by title."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(WikipediaArgs))
            .expect("failed to serialize wikipedia parameters schema")
    }

    fn requires_confirmation(&self) -> bool {
        false
    }

    async fn execute(&self, _args: &str, _ctx: &ExecutionContext) -> anyhow::Result<String> {
        anyhow::bail!("Wikipedia service is temporarily unavailable")
    }
}
