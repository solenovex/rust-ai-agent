use async_openai::types::chat::ChatCompletionTools;

use crate::tools::{
    calculator::definition::calculator_tool_definition,
    web_search::definition::web_search_tool_definition,
};

pub mod calculator;
pub mod web_search;

pub fn tools() -> Vec<ChatCompletionTools> {
    vec![calculator_tool_definition(), web_search_tool_definition()]
}
