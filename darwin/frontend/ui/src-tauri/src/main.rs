#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use directories::ProjectDirs;
use serde_json::json;
use std::{fs, path::PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
enum CfgError {
    #[error("app config dir not found")] 
    NoDir,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("keyring error: {0}")]
    Keyring(String),
}

fn config_path() -> Result<PathBuf, CfgError> {
    let proj = ProjectDirs::from("br", "darwin", "darwin-ragpp").ok_or(CfgError::NoDir)?;
    let dir = proj.config_dir().to_path_buf();
    if !dir.exists() { fs::create_dir_all(&dir)?; }
    Ok(dir.join("darwin_config.json"))
}

fn keyring_entry() -> Result<keyring::Entry, CfgError> {
    let entry = keyring::Entry::new("darwin-ragpp", "api-key").map_err(|e| CfgError::Keyring(e.to_string()))?;
    Ok(entry)
}

#[tauri::command]
fn cfg_load() -> Result<String, String> {
    (|| -> Result<String, CfgError> {
        let path = config_path()?;
        let mut data = if path.exists() {
            let txt = fs::read_to_string(&path)?;
            serde_json::from_str::<serde_json::Value>(&txt).unwrap_or(json!({}))
        } else {
            json!({})
        };

        if let Ok(entry) = keyring_entry() {
            if let Ok(secret) = entry.get_password() {
                data["DARWIN_SERVER_KEY"] = json!(secret);
            }
        }

        Ok(data.to_string())
    })()
    .map_err(|e| e.to_string())
}

#[tauri::command]
fn cfg_save(json_payload: String) -> Result<(), String> {
    (|| -> Result<(), CfgError> {
        let mut data: serde_json::Value = serde_json::from_str(&json_payload)?;

        // handle secret via OS keychain when present
        if let Some(secret) = data.get("DARWIN_SERVER_KEY").and_then(|v| v.as_str()) {
            if let Ok(entry) = keyring_entry() {
                // ignore set errors softly
                let _ = entry.set_password(secret);
            }
            // do not persist secret to disk
            if let Some(obj) = data.as_object_mut() { obj.remove("DARWIN_SERVER_KEY"); }
        }

        let path = config_path()?;
        fs::write(path, serde_json::to_vec_pretty(&data)?)?;
        Ok(())
    })()
    .map_err(|e| e.to_string())
}

#[tauri::command]
async fn secure_fetch(endpoint: String, method: String, body_json: Option<String>) -> Result<String, String> {
    (|| async move {
        // Load URL from config file
        let path = config_path()?;
        let base_url = if path.exists() {
            let txt = fs::read_to_string(&path)?;
            let v: serde_json::Value = serde_json::from_str(&txt)?;
            v.get("DARWIN_URL").and_then(|x| x.as_str()).map(|s| s.to_string())
                .or_else(|| v.get("NEXT_PUBLIC_DARWIN_URL").and_then(|x| x.as_str()).map(|s| s.to_string()))
                .ok_or_else(|| CfgError::Io(std::io::Error::new(std::io::ErrorKind::Other, "DARWIN_URL missing")))?
        } else {
            return Err(CfgError::Io(std::io::Error::new(std::io::ErrorKind::Other, "config not found")));
        };

        // Load API key from keyring
        let mut api_key: Option<String> = None;
        if let Ok(entry) = keyring_entry() {
            if let Ok(secret) = entry.get_password() {
                api_key = Some(secret);
            }
        }

        let url = if endpoint.starts_with("http") { endpoint } else { format!("{}{}{}", base_url.trim_end_matches('/'), if endpoint.starts_with('/') { "" } else { "/" }, endpoint) };
        let client = reqwest::Client::builder().build().map_err(|e| CfgError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        let mut req = match method.to_uppercase().as_str() {
            "GET" => client.get(&url),
            "POST" => client.post(&url),
            "PUT" => client.put(&url),
            "DELETE" => client.delete(&url),
            _ => client.post(&url),
        };
        req = req.header("Content-Type", "application/json");
        if let Some(k) = api_key { req = req.header("X-API-KEY", k); }
        if let Some(b) = body_json.clone() { req = req.body(b); }
        let resp = req.send().await.map_err(|e| CfgError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| CfgError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        if !status.is_success() {
            return Err(CfgError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("status {}: {}", status, text))));
        }
        Ok(text)
    })().await.map_err(|e: CfgError| e.to_string())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .invoke_handler(tauri::generate_handler![cfg_load, cfg_save, secure_fetch])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
