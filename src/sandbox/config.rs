use std::path::PathBuf;

/// Enables code execution. Only `podman` is supported; unset or `off` disables it.
pub const CODE_EXECUTION_ENV: &str = "CODE_EXECUTION";
/// Host directory mounted as /workspace in the sandbox (default `./workspace`).
pub const SANDBOX_WORKSPACE_ENV: &str = "SANDBOX_WORKSPACE";
/// Only files under this host directory may be uploaded. Unset means no upload tool.
pub const UPLOAD_ROOT_ENV: &str = "UPLOAD_ROOT";
/// Host folder of skills, mounted read-only at /skills in the sandbox. Optional.
pub const SKILLS_DIR_ENV: &str = "SKILLS_DIR";

#[derive(Debug, Clone, PartialEq)]
pub struct CodeExecutionSettings {
    pub workspace: PathBuf,
    pub upload_root: Option<PathBuf>,
    pub skills_dir: Option<PathBuf>,
}

/// Reads the settings through `get` (so tests need no real environment).
/// `Ok(None)` means code execution is disabled.
pub fn settings_from(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<CodeExecutionSettings>, String> {
    match get(CODE_EXECUTION_ENV).as_deref().map(str::trim) {
        None | Some("") | Some("off") | Some("disabled") => Ok(None),
        Some("podman") => Ok(Some(CodeExecutionSettings {
            workspace: PathBuf::from(
                get(SANDBOX_WORKSPACE_ENV).unwrap_or_else(|| "./workspace".to_string()),
            ),
            upload_root: get(UPLOAD_ROOT_ENV).map(PathBuf::from),
            skills_dir: get(SKILLS_DIR_ENV).map(PathBuf::from),
        })),
        Some(other) => Err(format!(
            "unsupported {CODE_EXECUTION_ENV} value: {other} (expected podman or off)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| map.get(k).cloned()
    }

    #[test]
    fn disabled_when_unset() {
        assert_eq!(settings_from(lookup(&[])).unwrap(), None);
    }

    #[test]
    fn explicit_off_values_are_disabled() {
        for v in ["", "off", "disabled"] {
            assert_eq!(
                settings_from(lookup(&[(CODE_EXECUTION_ENV, v)])).unwrap(),
                None,
                "value {v:?}"
            );
        }
    }

    #[test]
    fn podman_enables_with_default_workspace_and_no_upload_root() {
        let s = settings_from(lookup(&[(CODE_EXECUTION_ENV, "podman")]))
            .unwrap()
            .unwrap();
        assert_eq!(s.workspace, PathBuf::from("./workspace"));
        assert_eq!(s.upload_root, None);
    }

    #[test]
    fn custom_workspace_and_upload_root_are_honoured() {
        let s = settings_from(lookup(&[
            (CODE_EXECUTION_ENV, "podman"),
            (SANDBOX_WORKSPACE_ENV, "/data/ws"),
            (UPLOAD_ROOT_ENV, "/data/inputs"),
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(s.workspace, PathBuf::from("/data/ws"));
        assert_eq!(s.upload_root, Some(PathBuf::from("/data/inputs")));
    }

    #[test]
    fn skills_dir_is_optional_and_mounted_when_set() {
        let none = settings_from(lookup(&[(CODE_EXECUTION_ENV, "podman")]))
            .unwrap()
            .unwrap();
        assert_eq!(none.skills_dir, None);
        let some = settings_from(lookup(&[
            (CODE_EXECUTION_ENV, "podman"),
            (SKILLS_DIR_ENV, "/data/skills"),
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(some.skills_dir, Some(PathBuf::from("/data/skills")));
    }

    #[test]
    fn unknown_backend_is_a_clear_error() {
        let err = settings_from(lookup(&[(CODE_EXECUTION_ENV, "e2b")])).unwrap_err();
        assert!(err.contains("unsupported"), "{err}");
    }
}
