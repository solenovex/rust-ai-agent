use std::marker::PhantomData;

use schemars::{JsonSchema, schema_for};
use serde::{Serialize, de::DeserializeOwned};

use crate::{agent::context::ExecutionContext, constant::FINAL_ANSWER_TOOL_NAME, tool::Tool};

pub struct FinalAnswerTool<T>(PhantomData<fn() -> T>);

impl<T> FinalAnswerTool<T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

#[async_trait::async_trait]
impl<T> Tool for FinalAnswerTool<T>
where
    T: Serialize + DeserializeOwned + JsonSchema,
{
    fn name(&self) -> &str {
        FINAL_ANSWER_TOOL_NAME
    }

    fn description(&self) -> &str {
        "Return the final structured answer matching the required schema."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(T)).expect("Failed to serialize output schema")
    }

    fn requires_confirmation(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let output: T = serde_json::from_str(args_json)?;
        Ok(serde_json::to_string(&final_answer(output))?)
    }
}

impl<T> Default for FinalAnswerTool<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub fn final_answer<T>(output: T) -> T {
    output
}
