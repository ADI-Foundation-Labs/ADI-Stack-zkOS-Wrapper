//! Shivini-side glue for `GpuContextConfig`.
//!
//! The override channel is the `ZKOS_WRAPPER_MAX_DEVICE_ALLOCATION` environment
//! variable, read lazily at each `ProverContext::create*` call site. Wrapping it in
//! these helpers keeps the env-var read in one place; the wrapper CLI and downstream
//! prover services set the env var early in `main()` from their own `--memory-limit` /
//! `--max-device-allocation` flag.

use shivini::ProverContextConfig;

use crate::GpuContextConfig;

/// Layer the env-var-derived `max_device_allocation` onto an existing shivini config.
/// Callers that need other tweaks (e.g. `with_smallest_supported_domain_size` for the
/// compression circuit) build their base config first and then call this.
pub fn apply_env_overrides(base: ProverContextConfig) -> ProverContextConfig {
    match GpuContextConfig::from_env().max_device_allocation {
        Some(bytes) => base.with_maximum_device_allocation(bytes),
        None => base,
    }
}

/// Build a shivini config with env-var overrides applied to the defaults.
pub fn build_prover_context_config() -> ProverContextConfig {
    apply_env_overrides(ProverContextConfig::default())
}
