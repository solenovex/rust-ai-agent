use anyhow::Ok;

use crate::gaia::models::{GaiaRow, HfResponse};

pub async fn load_gaia_level1() -> anyhow::Result<Vec<GaiaRow>> {
    let token = std::env::var("HF_TOKEN")?;
    let client = reqwest::Client::new();
    let response = client
        .get("https://datasets-server.huggingface.co/rows")
        .query(&[
            ("dataset", "gaia-benchmark/GAIA"),
            ("config", "2023_level1"),
            ("split", "validation"),
            ("offset", "0"),
            ("length", "10"),
        ])
        .bearer_auth(token)
        .send()
        .await?
        .json::<HfResponse>()
        .await?;
    Ok(response.rows.into_iter().map(|r| r.row).collect())
}
