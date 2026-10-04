use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::Value;

use crate::{agent::context::ExecutionContext, tool::Tool};

pub struct CalculatorTool;

#[async_trait::async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Perform basic arithmetic operations: add (+), subtract (-), multiply (*), divide (/)."
    }

    fn parameters(&self) -> Value {
        serde_json::to_value(schema_for!(CalculatorArgs))
            .expect("Failed to serialize CalculatorArgs schema")
    }

    fn requires_confirmation(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: CalculatorArgs = serde_json::from_str(args_json)?;
        let result = calculator(&args.operator, args.first_number, args.second_number);
        match result {
            Ok(value) => Ok(value.to_string()),
            Err(err) => Ok(format!("Error: {err}")),
        }
    }
}

#[derive(Deserialize, Debug, JsonSchema)]
pub struct CalculatorArgs {
    pub operator: String,
    pub first_number: f64,
    pub second_number: f64,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CalcError {}

pub fn calculator(
    operator: &str,
    first_number: f64,
    second_number: f64,
) -> anyhow::Result<f64, String> {
    match operator {
        "add" | "+" => Ok(first_number + second_number),
        "subtract" | "-" => Ok(first_number - second_number),
        "multiply" | "*" => Ok(first_number * second_number),
        "divide" | "/" => {
            if second_number == 0.0 {
                Err("Cannot divide by zero".to_string())
            } else {
                Ok(first_number / second_number)
            }
        }
        other => Err(format!("Unsupported operator: {other}")),
    }
}
