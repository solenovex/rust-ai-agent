use std::sync::Arc;

use async_openai::types::chat::{
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs, FinishReason, ResponseFormat, ResponseFormatJsonSchema,
};
use backon::{ExponentialBuilder, Retryable};
use uuid::Uuid;

use crate::{
    agent::model::Agent,
    gaia::{
        dataset::get_cache_dir,
        model::{GaiaOutput, GaiaRow},
    },
    tool::ToolBox,
};

pub async fn solve_gaia_problem_with_retry(
    model: &str,
    system_instruction: &str,
    prompt: &str,
) -> anyhow::Result<GaiaOutput> {
    let op = || async { solve_problem(model, system_instruction, prompt).await };
    op.retry(ExponentialBuilder::default().with_max_times(3))
        .await
}

async fn solve_problem(
    model: &str,
    system_instruction: &str,
    user_prompt: &str,
) -> anyhow::Result<GaiaOutput> {
    let schema = schemars::schema_for!(GaiaOutput);
    let schema_json = serde_json::to_value(&schema)?;
    let response_format_setting = ResponseFormat::JsonSchema {
        json_schema: ResponseFormatJsonSchema {
            description: Some("GAIA problem solving output".into()),
            name: "gaia_output".into(),
            schema: schema_json,
            strict: Some(true),
        },
    };

    let client = async_openai::Client::new();
    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system_instruction)
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_prompt)
                .build()?
                .into(),
        ])
        .response_format(response_format_setting)
        .build()?;

    let response = client.chat().create(request).await?;

    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No choices in response"))?;

    if choice.finish_reason == Some(FinishReason::ContentFilter) {
        return Ok(GaiaOutput {
            is_solvable: false,
            unsolvable_reason: "Model refuse to answer".to_owned(),
            final_answer: String::new(),
        });
    }

    let content = choice
        .message
        .content
        .ok_or_else(|| anyhow::anyhow!("No content in response"))?;
    let output: GaiaOutput = serde_json::from_str(&content)?;

    Ok(output)
}

pub async fn solve_gaia_problem_with_tools_with_retry(
    model: &str,
    system_instruction: &str,
    problem: &GaiaRow,
    toolbox: Arc<ToolBox>,
) -> anyhow::Result<GaiaOutput> {
    let op = || async {
        solve_problem_with_tools(model, system_instruction, problem, toolbox.clone()).await
    };
    op.retry(ExponentialBuilder::default().with_max_times(3))
        .await
}

pub async fn solve_problem_with_tools(
    model: &str,
    system_instruction: &str,
    problem: &GaiaRow,
    toolbox: Arc<ToolBox>,
) -> anyhow::Result<GaiaOutput> {
    let agent = Agent::new(
        model.to_owned(),
        vec![system_instruction.to_owned()],
        toolbox,
    );
    let user_prompt = match problem.file_name.as_deref() {
        Some(file_name) => {
            let cache_dir = get_cache_dir()?;
            format!(
                "{}\nThe attached file is located at: {}",
                problem.question,
                cache_dir.join(file_name).display()
            )
        }
        None => problem.question.clone(),
    };
    let session_id = Uuid::new_v4().to_string();
    let agent_result = agent
        .run::<GaiaOutput>(&user_prompt, &session_id, None)
        .await?;
    Ok(agent_result.output.unwrap_or(GaiaOutput {
        is_solvable: false,
        unsolvable_reason: String::from("Agent returns no answer"),
        final_answer: String::default(),
    }))
}
