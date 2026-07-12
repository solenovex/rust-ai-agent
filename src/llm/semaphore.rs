use std::sync::OnceLock;

use tokio::sync::Semaphore;

static SEMAPHORE: OnceLock<Semaphore> = OnceLock::new();

pub fn get_semaphore() -> &'static Semaphore {
    SEMAPHORE.get_or_init(|| Semaphore::new(3))
}