use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObjectArgs};
use serde_json::json;

pub fn web_search_tool_definition() -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObjectArgs::default()
            .name("web_search")
            .description("Search the web for current information on a given query.")
            .parameters(json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute with Tavily"
                    },
                    "max_results": {
                        "type": "number",
                        "description": "The maximum number of search results to return",
                        "default": 5,
                        "minimum": 0,
                        "maximum": 20
                    },
                    "topic": {
                        "type": "string",
                        "description": "Use 'news' for recent events, sports, politics, or anything time-sensitive. Use 'general' for broad or evergreen questions.",
                        "enum": ["general", "news", "finance"],
                        "default": "general"
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Restrict results to a recent time window. Use when the query is about something recent or ongoing.",
                        "enum": ["day", "week", "month", "year"]
                    },
                },
                "required": ["query"]
            }))
            .build()
            .expect("failed to build web search tool definition"),
    })
}