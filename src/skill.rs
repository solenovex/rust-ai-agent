use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A skill: a folder containing a `SKILL.md`. Only `name` and `description` go into the prompt;
/// the full instructions are read on demand, which keeps the prompt small.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillInfo {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
}

/// Parses the `key: value` lines of a YAML frontmatter block delimited by `---`.
/// Only the first colon splits, so values may contain colons.
pub fn parse_frontmatter(content: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut lines = content.lines();
    if lines.next().map(str::trim) != Some("---") {
        return map;
    }
    for line in lines {
        if line.trim() == "---" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            let v = v.trim().trim_matches(|c| c == '"' || c == '\'');
            map.insert(k.trim().to_string(), v.to_string());
        }
    }
    map
}

/// Loads one skill folder. Returns `None` if `SKILL.md` is missing or lacks `name`/`description`.
pub fn load_skill(dir: &Path) -> Option<SkillInfo> {
    let content = std::fs::read_to_string(dir.join("SKILL.md")).ok()?;
    let fm = parse_frontmatter(&content);
    let name = fm.get("name").filter(|s| !s.is_empty())?.clone();
    let description = fm.get("description").filter(|s| !s.is_empty())?.clone();
    Some(SkillInfo {
        name,
        description,
        path: dir.to_path_buf(),
    })
}

/// Lists valid skills under `skills_path`, sorted by name. A missing directory yields none.
pub fn discover_skills(skills_path: &Path) -> Vec<SkillInfo> {
    let Ok(entries) = std::fs::read_dir(skills_path) else {
        return vec![];
    };
    let mut skills: Vec<SkillInfo> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter_map(|p| load_skill(&p))
        .collect();
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

/// Builds the system-prompt section: one line per skill plus how to read the full file.
/// `sandbox_path` is where the skills folder is mounted inside the container.
pub fn generate_skills_prompt(skills: &[SkillInfo], sandbox_path: &str) -> String {
    if skills.is_empty() {
        return String::new();
    }
    let mut lines = vec![
        "## Available Skills".to_string(),
        format!(
            "Skills are installed at `{sandbox_path}`. Read the SKILL.md file before using a skill."
        ),
        String::new(),
    ];
    for s in skills {
        lines.push(format!("- **{}**: {}", s.name, s.description));
    }
    lines.push(String::new());
    lines.push("To use a skill:".into());
    lines.push("```bash".into());
    lines.push(format!("cat {sandbox_path}/<skill-name>/SKILL.md"));
    lines.push("```".into());
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const PDF_MERGE: &str = "---\nname: pdf-merge\ndescription: Merge multiple PDF files into one. Use when: the user wants to combine PDFs.\n---\n\n# PDF Merge\n";

    #[test]
    fn parses_name_and_description_even_with_colons_in_the_value() {
        let fm = parse_frontmatter(PDF_MERGE);
        assert_eq!(fm["name"], "pdf-merge");
        assert!(fm["description"].contains("Use when: the user"));
    }

    #[test]
    fn no_frontmatter_means_empty_map() {
        assert!(parse_frontmatter("# just markdown").is_empty());
    }

    #[test]
    fn strips_matching_quotes_around_values() {
        let fm = parse_frontmatter("---\nname: \"quoted\"\ndescription: 'single'\n---\n");
        assert_eq!(fm["name"], "quoted");
        assert_eq!(fm["description"], "single");
    }

    #[test]
    fn discover_skips_invalid_folders_and_sorts_by_name() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("pdf-merge")).unwrap();
        std::fs::write(root.path().join("pdf-merge/SKILL.md"), PDF_MERGE).unwrap();
        std::fs::create_dir_all(root.path().join("broken")).unwrap();
        std::fs::write(root.path().join("broken/SKILL.md"), "no frontmatter").unwrap();
        std::fs::create_dir_all(root.path().join("empty")).unwrap();
        std::fs::write(root.path().join("loose.txt"), "not a folder").unwrap();

        let skills = discover_skills(root.path());
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "pdf-merge");
        assert_eq!(skills[0].path, root.path().join("pdf-merge"));
    }

    #[test]
    fn discover_on_a_missing_directory_is_empty() {
        assert!(discover_skills(Path::new("/definitely/not/here")).is_empty());
    }

    #[test]
    fn prompt_lists_names_and_descriptions_with_a_read_instruction() {
        let s = SkillInfo {
            name: "pdf-merge".into(),
            description: "Merge PDFs".into(),
            path: "/x".into(),
        };
        let p = generate_skills_prompt(&[s], "/skills");
        assert!(p.contains("- **pdf-merge**: Merge PDFs"));
        assert!(p.contains("cat /skills/<skill-name>/SKILL.md"));
    }

    #[test]
    fn prompt_is_empty_without_skills() {
        assert_eq!(generate_skills_prompt(&[], "/skills"), "");
    }
}
