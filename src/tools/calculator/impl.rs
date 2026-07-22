use schemars::schema_for;
use serde_json::Value;

use crate::tools::{
    calculator::execute::{CalculatorArgs, calculator},
    tool::Tool,
};

pub struct CalculatorTool;

#[async_trait::async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Perform basic arithmetic operations: add, subtract, multiply, divide."
    }

    fn parameters(&self) -> Value {
        serde_json::to_value(schema_for!(CalculatorArgs))
            .expect("Failed to serialize CalculatorArgs schema")
    }

    async fn execute(&self, args_json: &str) -> anyhow::Result<String> {
        let args: CalculatorArgs = serde_json::from_str(args_json)?;
        let result = calculator(&args.operator, args.first_number, args.second_number);
        match result {
            Ok(value) => Ok(value.to_string()),
            Err(err) => Ok(format!("Error: {err}")),
        }
    }
}
