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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ToolResultStatus;
    use serde_json::json;

    #[test]
    fn compacts_known_tool_results_but_preserves_unknown_tools() {
        let mut request = LlmRequest {
            instructions: Vec::new(),
            contents: vec![
                ContentItem::ToolCall(ToolCall {
                    tool_call_id: "read-1".to_string(),
                    name: "read_file".to_string(),
                    arguments: json!({"file_path": "notes.txt"}),
                }),
                ContentItem::ToolResult {
                    tool_call_id: "read-1".to_string(),
                    name: "read_file".to_string(),
                    status: ToolResultStatus::Success,
                    content: "large file contents".to_string(),
                },
                ContentItem::ToolResult {
                    tool_call_id: "other-1".to_string(),
                    name: "custom_tool".to_string(),
                    status: ToolResultStatus::Success,
                    content: "custom output".to_string(),
                },
            ],
        };

        Compaction { keep_recent: 1 }.apply(&mut request);

        let ContentItem::ToolResult { content, .. } = &request.contents[1] else {
            panic!("expected a tool result");
        };
        assert!(content.contains("notes.txt"));
        let ContentItem::ToolResult { content, .. } = &request.contents[2] else {
            panic!("expected a tool result");
        };
        assert_eq!(content, "custom output");
    }
}
