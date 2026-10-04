use std::path::PathBuf;

use crate::{
    constant::HUGGING_FACE_TOKEN_NAME,
    gaia::model::{GaiaRow, HfFile, HfResponse},
};

pub async fn load_gaia_level1() -> anyhow::Result<Vec<GaiaRow>> {
    let token = std::env::var(HUGGING_FACE_TOKEN_NAME)?;
    let client = reqwest::Client::new();
    let response = client
        .get("https://datasets-server.huggingface.co/rows")
        .query(&[
            ("dataset", "gaia-benchmark/GAIA"),
            ("config", "2023_level1"),
            ("split", "validation"),
            ("offset", "0"),
            ("length", "20"),
        ])
        .bearer_auth(token)
        .send()
        .await?
        .json::<HfResponse>()
        .await?;
    Ok(response.rows.into_iter().map(|r| r.row).collect())
}

pub async fn load_gaia_dataset() -> anyhow::Result<Vec<GaiaRow>> {
    let token = std::env::var(HUGGING_FACE_TOKEN_NAME)?;

    let mut all_rows = Vec::new();

    let client = reqwest::Client::new();
    let mut offset = 0usize;
    const PAGE_SIZE: usize = 100;

    loop {
        let response = client
            .get("https://datasets-server.huggingface.co/rows")
            .query(&[
                ("dataset", "gaia-benchmark/GAIA"),
                ("config", "2023_all"),
                ("split", "validation"),
                ("offset", &offset.to_string()),
                ("length", &PAGE_SIZE.to_string()),
            ])
            .bearer_auth(token.clone())
            .send()
            .await?
            .json::<HfResponse>()
            .await?;
        let count = response.rows.len();
        all_rows.extend(response.rows.into_iter().map(|r| r.row));
        if count < PAGE_SIZE {
            break;
        }
        offset += PAGE_SIZE;
    }

    Ok(all_rows)
}

pub fn get_cache_dir() -> anyhow::Result<PathBuf> {
    let project_root = std::env::current_dir()?;
    Ok(project_root.join("gaia_cache"))
}

#[allow(unused)]
fn remove_cache_dir() -> anyhow::Result<()> {
    let cache_dir = get_cache_dir()?;
    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir)?;
    }
    Ok(())
}

pub async fn download_gaia_dataset_attachments() -> anyhow::Result<()> {
    let token = std::env::var(HUGGING_FACE_TOKEN_NAME)?;

    let client = reqwest::Client::new();
    let files = client
        .get("https://huggingface.co/api/datasets/gaia-benchmark/GAIA/tree/main/2023/validation")
        .query(&[("recursive", "true")])
        .bearer_auth(&token)
        .send()
        .await?
        .json::<Vec<HfFile>>()
        .await?;

    println!("{} files to download", files.len());

    let cache_dir = get_cache_dir()?;

    for file in files.iter() {
        let local_file_path = cache_dir.join(&file.path);
        if let Some(parent) = local_file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        if local_file_path.exists() {
            println!("Already exists: {}", local_file_path.display());
            continue;
        }

        let url = format!(
            "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/{}",
            file.path
        );

        let bytes = client
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await?
            .bytes()
            .await?;

        tokio::fs::write(&local_file_path, bytes).await?;
        println!("Downloaded: {}", local_file_path.display());
    }

    Ok(())
}

pub async fn experiment_gaia_dataset() -> anyhow::Result<()> {
    let dataset = load_gaia_dataset().await?;
    download_gaia_dataset_attachments().await?;

    let problems_with_files: Vec<&GaiaRow> =
        dataset.iter().filter(|r| r.file_name.is_some()).collect();
    let problems_with_zips: Vec<&GaiaRow> = dataset
        .iter()
        .filter(|r| r.file_name.clone().is_some_and(|r| r.ends_with(".zip")))
        .collect();

    println!("Total problems: {}", dataset.len());
    println!("Problem with attachments: {}", problems_with_files.len());
    println!("Problem with zips: {}", problems_with_zips.len());

    if !problems_with_zips.is_empty() {
        let zip_problem = problems_with_zips[0];
        println!(
            "Question: {}",
            zip_problem.question.chars().take(100).collect::<String>()
        );
        println!(
            "File name: {}",
            zip_problem.file_name.as_deref().unwrap_or_default()
        );
        let cache_dir = get_cache_dir()?;
        let local_file_path = cache_dir.join(zip_problem.file_path.as_deref().unwrap_or_default());
        println!("File exists: {}", local_file_path.exists());
    }

    Ok(())
}
