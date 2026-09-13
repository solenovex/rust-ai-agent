use std::collections::HashMap;

use serde_json::Value;

use crate::agent::{ContentItem, event::ToolCall, llm_request::LlmRequest};

pub struct Compaction {
    pub keep_recent: usize,
}

impl Compaction {
    pub fn apply(&self, request: &mut LlmRequest) {
        let protect_from = request.contents.len().saturating_sub(self.keep_recent);
        let mut call_args: HashMap<String, Value> = HashMap::new();

        for (idx, item) in request.contents.iter_mut().enumerate() {
            match item {
                ContentItem::ToolCall(ToolCall {
                    tool_call_id,
                    arguments,
                    ..
                }) => {
                    call_args.insert(tool_call_id.clone(), arguments.clone());
                }
                ContentItem::ToolResult {
                    tool_call_id,
                    name,
                    content,
                    ..
                } => {
                    if idx >= protect_from {
                        continue;
                    }

                    let args = call_args.get(tool_call_id);
                    let replacement = match name.as_str() {
                        "read_file" => {
                            let path = args
                                .and_then(|a| a.get("file_path"))
                                .and_then(Value::as_str)
                                .unwrap_or("unknown");

                            Some(format!(
                                "File '{path}' was already read. Call read_file again if you need it."
                            ))
                        }
                        "web_search" => {
                            let query = args
                                .and_then(|a| a.get("query"))
                                .and_then(Value::as_str)
                                .unwrap_or("unknown");
                            Some(format!(
                                "Search results for '{query}' were already processed. Call web_search again if you need them."
                            ))
                        }
                        _ => None,
                    };

                    if let Some(replacement) = replacement {
                        *content = replacement;
                    }
                }
                _ => {}
            }
        }
    }
}
