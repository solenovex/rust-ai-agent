use std::io::{self, Write};

use async_openai::types::chat::ChatCompletionMessageToolCall;

use crate::agent::{callback::before_tool_callback::BeforeToolCallback, context::ExecutionContext};

pub struct ApprovalCallback;

const DANGEROUS_TOOLS: &[&str] = &["delete_file", "send_email", "execute_sql"];

#[async_trait::async_trait]
impl BeforeToolCallback for ApprovalCallback {
    async fn call(
        &self,
        _context: &ExecutionContext,
        tool_call: &ChatCompletionMessageToolCall,
    ) -> Option<String> {
        if DANGEROUS_TOOLS.contains(&tool_call.function.name.as_str()) {
            println!("\n Dangerous tool execution requested");
            println!("Tool: {}", tool_call.function.name);
            println!("Arguments: {}", tool_call.function.arguments);

            println!("Do you want to execute? (y/n)");
            io::stdout().flush().unwrap();
            let mut response = String::new();
            io::stdin().read_line(&mut response).unwrap();

            let response = response.trim().to_lowercase();
            match response.as_str() {
                "y" => {
                    println!("Approved. Executing...\n");
                    None
                }
                "n" => {
                    println!("Denied. Skipping execution.\n");
                    Some(format!(
                        "User denied execution of {}",
                        tool_call.function.name
                    ))
                }
                _ => Some(format!(
                    "Tool execution was not approved because the user entered an invalid response: '{}'.",
                    response
                )),
            }
        } else {
            None
        }
    }
}
