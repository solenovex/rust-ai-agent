use std::path::{Component, Path, PathBuf};

use crate::sandbox::SandboxError;

/// Value object: a path guaranteed to stay inside the workspace
/// (relative, no `..`, no root component).
#[derive(Debug, Clone, PartialEq)]
pub struct WorkspacePath(PathBuf);

impl WorkspacePath {
    pub fn parse(raw: &str) -> Result<Self, SandboxError> {
        let invalid = || SandboxError::InvalidPath(raw.to_string());
        let p = Path::new(raw);
        let p = if p.is_absolute() {
            p.strip_prefix("/workspace").map_err(|_| invalid())?
        } else {
            p
        };
        if p.as_os_str().is_empty() || p.components().any(|c| !matches!(c, Component::Normal(_))) {
            return Err(invalid());
        }
        Ok(Self(p.to_path_buf()))
    }

    pub fn as_path(&self) -> &Path {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn rejects_parent_dir() {
        assert!(WorkspacePath::parse("../etc/passwd").is_err());
        assert!(WorkspacePath::parse("a/../../b").is_err());
    }

    #[test]
    fn rejects_absolute_outside_workspace() {
        assert!(WorkspacePath::parse("/etc/passwd").is_err());
    }

    #[test]
    fn accepts_relative_and_workspace_prefixed() {
        assert_eq!(
            WorkspacePath::parse("data.xlsx").unwrap().as_path(),
            Path::new("data.xlsx")
        );
        assert_eq!(
            WorkspacePath::parse("/workspace/out/a.csv")
                .unwrap()
                .as_path(),
            Path::new("out/a.csv")
        );
    }

    #[test]
    fn rejects_empty() {
        assert!(WorkspacePath::parse("").is_err());
        assert!(WorkspacePath::parse("/workspace").is_err());
    }
}
