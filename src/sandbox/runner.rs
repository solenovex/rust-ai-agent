/// The container-side resident Python runner, embedded at compile time.
pub const RUNNER_PY: &str = include_str!("runner.py");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runner_speaks_the_json_line_protocol() {
        assert!(RUNNER_PY.contains("def main():"));
        assert!(
            RUNNER_PY.contains("\"type\": \"exec\"")
                || RUNNER_PY.contains("req[\"type\"] == \"exec\"")
        );
        assert!(RUNNER_PY.contains("\"type\": \"result\""));
    }

    #[test]
    fn runner_exposes_the_host_tool_bridge() {
        assert!(RUNNER_PY.contains("def _call_tool(tool_name, args):"));
        assert!(RUNNER_PY.contains("\"type\": \"tool_call\""));
    }

    #[test]
    fn timeout_is_base_exception_so_user_code_cannot_swallow_it() {
        assert!(RUNNER_PY.contains("class _Timeout(BaseException):"));
    }
}
