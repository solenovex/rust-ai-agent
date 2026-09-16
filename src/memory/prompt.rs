pub const TASK_MEMORY_EXTRACTION_PROMPT: &str = r#"You are analyzing an agent's problem-solving record to extract a learning record.

Analyze the following execution history:

<execution_history>
{execution_history}
</execution_history>

Fill in each field precisely:
- task_summary: a short, neutral description of what the user asked — the problem itself, not the outcome.
- approach: only the method/formula/tool the agent actually used to attempt the problem. Do not describe your own analysis process, and do not mention whether it succeeded or failed here.
- final_answer: the answer the agent actually gave (or the corrected answer, if the user provided one).
- is_correct: true only if the agent's answer was correct; false if the agent made a mistake, even if it was later corrected.
- error_analysis: if is_correct is false, explain concretely what went wrong (e.g. used the wrong formula, misread the input) — this field must not be empty when is_correct is false. If is_correct is true, leave this field null.
"#;

pub const DUPLICATE_CHECK_PROMPT: &str = r#"Compare the new memory against existing memories to determine if it's a duplicate.

Existing memories:
{existing_memories}

New memory:
{new_memory}

Respond with one of:
- ADD: This is new information that should be stored
- SKIP: Similar information already exists, no need to store

Judgment criteria:
- Same problem with different approach or different result counts as new information
- Same problem with same approach and same result is a duplicate
"#;