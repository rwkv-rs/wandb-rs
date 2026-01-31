use std::{collections::HashMap, time::Duration};

use reqwest::StatusCode;
use serde::Serialize;
use tokio::{sync::mpsc, task::JoinHandle};
use tracing::{error, info, warn};

use crate::{data_value::LogData, ApiError, ReqwestBadResponse};

const LOG_BATCH_SIZE: usize = 128;
const LOG_RETRY_MAX: usize = 5;
const LOG_RETRY_BASE_DELAY_MS: u64 = 200;
const LOG_RETRY_MAX_DELAY_MS: u64 = 5_000;


pub struct Run {
    tx_log_data: Option<mpsc::Sender<RunMessage>>,
    log_thread: Option<JoinHandle<Result<(), ApiError>>>,
}

#[derive(Debug, Serialize)]
struct FsChunkData {
    content: Vec<String>,
    offset: u64,
}

#[derive(Debug, Serialize)]
struct FsFilesData {
    files: HashMap<String, FsChunkData>,
}

async fn submit_log(
    client: &reqwest::Client,
    run_path: &str,
    step: u64,
    rows: &[LogData],
) -> Result<(), ApiError> {
    if rows.is_empty() {
        return Ok(());
    }

    let mut content = Vec::with_capacity(rows.len());
    for row in rows {
        let mut line = serde_json::to_string(row)?;
        line.push('\n');
        content.push(line);
    }

    let summary_row = serde_json::to_string(
        rows.last().expect("rows should not be empty"),
    )?;

    let log = FsFilesData {
        files: [
            (
                "wandb-history.jsonl".to_string(),
                FsChunkData {
                    content,
                    offset: step,
                },
            ),
            (
                "wandb-summary.json".to_string(),
                FsChunkData {
                    content: vec![summary_row],
                    offset: 0,
                },
            ),
        ]
        .into_iter()
        .collect(),
    };

    let mut retries = 0;
    loop {
        let response = client.post(run_path).json(&log).send().await;
        match response {
            Ok(response) => match response.maybe_err().await {
                Ok(_) => return Ok(()),
                Err(err) => {
                    let api_error = ApiError::from(err);
                    if retries >= LOG_RETRY_MAX || !should_retry(&api_error) {
                        return Err(api_error);
                    }
                    let delay = retry_delay(retries);
                    warn!(
                        "Log batch upload failed (attempt {}/{}), retrying in {:?}: {}",
                        retries + 1,
                        LOG_RETRY_MAX,
                        delay,
                        api_error
                    );
                    tokio::time::sleep(delay).await;
                }
            },
            Err(err) => {
                let api_error = ApiError::from(err);
                if retries >= LOG_RETRY_MAX || !should_retry(&api_error) {
                    return Err(api_error);
                }
                let delay = retry_delay(retries);
                warn!(
                    "Log batch upload failed (attempt {}/{}), retrying in {:?}: {}",
                    retries + 1,
                    LOG_RETRY_MAX,
                    delay,
                    api_error
                );
                tokio::time::sleep(delay).await;
            }
        }
        retries += 1;
    }
}

fn should_retry(error: &ApiError) -> bool {
    match error {
        ApiError::RequestErrorWithBody(err) => should_retry_reqwest_error(&err.error),
        ApiError::RequestFailed(err) => should_retry_reqwest_error(err),
        _ => false,
    }
}

fn should_retry_reqwest_error(error: &reqwest::Error) -> bool {
    if error.is_timeout() || error.is_connect() {
        return true;
    }
    error
        .status()
        .map(|status| status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error())
        .unwrap_or(false)
}

fn retry_delay(retries: usize) -> Duration {
    let exp = retries.min(6) as u32;
    let delay_ms = LOG_RETRY_BASE_DELAY_MS.saturating_mul(1u64 << exp);
    Duration::from_millis(delay_ms.min(LOG_RETRY_MAX_DELAY_MS))
}

enum RunMessage {
    // TODO: add FinishRun
    LogData(LogData),
}

impl Run {
    pub fn new(
        base_url: String,
        client: reqwest::Client,
        entity: String,
        project: String,
        name: String,
    ) -> Run {
        let (tx_log_data, mut rx_log_data) = mpsc::channel::<RunMessage>(512);
        let log_thread: JoinHandle<Result<(), ApiError>> = tokio::spawn(async move {
            let run_path = format!("{base_url}/files/{entity}/{project}/{name}/file_stream");
            let mut step = 0;
            let mut buffer: Vec<LogData> = Vec::with_capacity(LOG_BATCH_SIZE);
            while let Some(message) = rx_log_data.recv().await {
                match message {
                    RunMessage::LogData(row) => {
                        buffer.push(row);
                        if buffer.len() >= LOG_BATCH_SIZE {
                            let batch_len = buffer.len() as u64;
                            if let Err(log_error) =
                                submit_log(&client, &run_path, step, &buffer).await
                            {
                                error!(
                                    "Failed to log batch to WandB for step {step}: {log_error}"
                                );
                            } else {
                                step += batch_len;
                                buffer.clear();
                            }
                        }
                    }
                }
            }
            if !buffer.is_empty() {
                let batch_len = buffer.len() as u64;
                if let Err(log_error) = submit_log(&client, &run_path, step, &buffer).await {
                    error!("Failed to log batch to WandB for step {step}: {log_error}");
                } else {
                    step += batch_len;
                    buffer.clear();
                }
            }
            info!("WandB run {name} ended.");
            Ok(())
        });
        Run {
            tx_log_data: Some(tx_log_data),
            log_thread: Some(log_thread),
        }
    }

    pub async fn finish(mut self) -> Result<(), ApiError> {
        let log_thread = self.log_thread.take();
        drop(self.tx_log_data.take());
        if let Some(log_thread) = log_thread {
            log_thread.await??;
        }
        Ok(())
    }

    /// Upload run data.

    /// Use `log` to log data from runs, such as scalars, images, vereo,
    /// histograms, plots, and tables.
    ///
    /// See our [guides to logging](https://docs.wandb.ai/guides/track/log) for
    /// live examples, code snippets, best practices, and more.
    ///
    /// The most basic usage is `run.log(("train-loss", 0.5), ("accuracy", 0.9))`.
    /// This will save the loss and accuracy to the run's history and update
    /// the summary values for these metrics.
    ///
    /// Visualize logged data in the workspace at [wandb.ai](https://wandb.ai),
    /// or locally on a [self-hosted instance](https://docs.wandb.ai/guides/hosting)
    /// of the W&B app, or export data to visualize and explore locally, e.g. in
    /// Jupyter notebooks, with [our API](https://docs.wandb.ai/guides/track/public-api-guide).
    ///
    /// Logged values don't have to be scalars. Logging any wandb object is supported.
    /// For example `run.log({"example": wandb.Image("myimage.jpg")})` will log an
    /// example image which will be displayed nicely in the W&B UI.
    /// See the [reference documentation](https://docs.wandb.com/ref/python/data-types)
    /// for all of the different supported types or check out our
    /// [guides to logging](https://docs.wandb.ai/guides/track/log) for examples,
    /// from 3D molecular structures and segmentation masks to PR curves and histograms.
    /// You can use `wandb.Table` to log structured data. See our
    /// [guide to logging tables](https://docs.wandb.ai/guides/data-vis/log-tables)
    /// for details.
    ///
    /// The W&B UI organizes metrics with a forward slash (`/`) in their name
    /// into sections named using the text before the final slash. For example,
    /// the following results in two sections named "train" and "validate":
    ///
    /// ```
    /// run.log((
    ///     ("train/accuracy", 0.9),
    ///     ("train/loss", 30),
    ///     ("validate/accuracy", 0.8),
    ///     ("validate/loss", 20),
    /// ));
    /// ```
    ///
    /// Only one level of nesting is supported; `run.log({"a/b/c": 1})`
    /// produces a section named "a/b".
    ///
    /// `run.log` is not intended to be called more than a few times per second.
    /// For optimal performance, limit your logging to once every N iterations,
    /// or collect data over multiple iterations and log it in a single step.
    ///
    /// ### The W&B step
    ///
    /// With basic usage, each call to `log` creates a new "step".
    /// The step must always increase, and it is not possible to log
    /// to a previous step.
    ///
    /// Note that you can use any metric as the X axis in charts.
    /// In many cases, it is better to treat the W&B step like
    /// you'd treat a timestamp rather than a training step.
    ///
    /// ```
    /// // Example: log an "epoch" metric for use as an X axis.
    /// run.log((
    ///     ("epoch", 40),
    ///     ("train-loss", 0.5)
    /// ));
    /// ```
    /// See also [define_metric](https://docs.wandb.ai/ref/python/run#define_metric).
    pub async fn log(&self, row: impl Into<LogData>) {
        // hack to prevent nasty monomorphization blowup -
        // only the .into() is monomorphized, the rest is not.
        self._log(row.into()).await
    }
    async fn _log(&self, row: LogData) {
        if let Some(tx_log_data) = self.tx_log_data.as_ref() {
            if let Err(e) = tx_log_data.send(RunMessage::LogData(row)).await {
                warn!("Failed to send log data to wandb: {}", e);
            }
        } else {
            warn!("Failed to send log data to wandb: logger already finished");
        }
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        drop(self.tx_log_data.take());
        if let Some(log_thread) = self.log_thread.take() {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    match log_thread.await {
                        Ok(Ok(())) => {}
                        Ok(Err(err)) => {
                            error!("WandB log thread failed: {err}");
                        }
                        Err(err) => {
                            error!("WandB log thread panicked: {err}");
                        }
                    }
                });
            } else {
                log_thread.abort();
            }
        }
    }
}
