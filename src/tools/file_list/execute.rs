use std::{fs, path::Path};

pub fn list(path: &str) -> anyhow::Result<String> {
    let path = Path::new(path);
    if !path.exists() {
        anyhow::bail!("Path not found: {}", path.display());
    }
    if !path.is_dir() {
        anyhow::bail!("Not a directory: {}", path.display());
    }

    let mut dirs = Vec::new();
    let mut files = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if entry.file_type()?.is_dir() {
            dirs.push(format!("{name}/"));
        } else {
            files.push(name);
        }
    }
    dirs.sort();
    files.sort();

    let mut result = format!("Directory: {}\n", path.display());
    for item in dirs.into_iter().chain(files) {
        result.push_str(&format!(". {item}\n"));
    }

    Ok(result)
}
