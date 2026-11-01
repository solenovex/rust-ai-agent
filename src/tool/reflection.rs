use anyhow::Context;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};

use crate::{agent::context::ExecutionContext, tool::Tool};

pub struct ReflectionTool;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ReflectionArgs {
    /// Your assessment of the current situation and next direction.
    pub analysis: String,

    /// Set to true if the current plan needs modification.
    #[serde(default)]
    pub need_replan: bool,
}

#[async_trait::async_trait]
impl Tool for ReflectionTool {
    fn name(&self) -> &str {
        "reflection"
    }

    fn description(&self) -> &str {
        r#"Pause and analyze progress before continuing.

WHEN TO USE:
1. PROGRESS REVIEW - After completing a meaningful step
   "Kipchoge's record found: 2:01:09. Moving to moon distance research."
2. ERROR ANALYSIS - When a tool fails or returns unexpected results
   "Wikipedia tool failed. Cause: service unavailable. Alternative: use web search."
3. RESULT SYNTHESIS - When combining information from multiple sources
   "Two different moon distances found. Problem asks for closest approach, so using perigee: 356,500km."
4. SELF CHECK - Before providing final answer
   "Have all required data: marathon pace 20.81km/h, moon distance 356,500km. Ready to calculate."

WHEN NOT TO USE:
- After every single tool call (excessive overhead)
- During simple, straightforward operations
- When everything is proceeding as expected"#
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(ReflectionArgs))
            .expect("Failed to serialize reflection parameters schema")
    }

    fn requires_confirmation(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: ReflectionArgs =
            serde_json::from_str(args_json).context("invalid reflection arguments")?;
        let result = reflection(&args.analysis, args.need_replan);
        Ok(result)
    }
}

pub fn reflection(analysis: &str, need_replan: bool) -> String {
    if need_replan {
        format!("Reflection recorded (REPLAN NEEDED): {analysis}")
    } else {
        format!("Reflection recorded: {analysis}")
    }
}

#[cfg(test)]
mod reflection_tool_tests {
    use super::*;

    #[test]
    fn need_replan_defaults_to_false() {
        let args: ReflectionArgs = serde_json::from_str(r#"{"analysis":"ok"}"#).unwrap();
        assert!(!args.need_replan);
    }

    #[test]
    fn schema_does_not_require_need_replan() {
        let schema = ReflectionTool.parameters();
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().all(|v| v != "need_replan"));
    }
}
