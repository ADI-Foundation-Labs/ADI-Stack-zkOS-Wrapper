//! User-facing GPU context configuration.
//!
//! `GpuContextConfig` is plumbed through every public proving / setup function so callers
//! can cap shivini's device memory pool. Lives outside the `gpu` feature gate so the
//! field can also be threaded through non-GPU builds without `#[cfg]` noise on every fn.

pub const MAX_DEVICE_ALLOCATION_ENV: &str = "ZKOS_WRAPPER_MAX_DEVICE_ALLOCATION";

#[derive(Copy, Clone, Debug, Default)]
pub struct GpuContextConfig {
    /// Maps to `shivini::ProverContextConfig::with_maximum_device_allocation`.
    /// `None` lets shivini grab all free device memory at startup (the historical default).
    pub max_device_allocation: Option<usize>,
}

impl GpuContextConfig {
    pub fn with_max_device_allocation(mut self, bytes: usize) -> Self {
        self.max_device_allocation = Some(bytes);
        self
    }

    /// Reads `ZKOS_WRAPPER_MAX_DEVICE_ALLOCATION` if set. Returns the default config when
    /// the env var is absent. Panics on parse errors so misconfiguration fails fast at
    /// startup instead of being silently ignored.
    pub fn from_env() -> Self {
        match std::env::var(MAX_DEVICE_ALLOCATION_ENV) {
            Ok(raw) => {
                let bytes = parse_byte_size(&raw).unwrap_or_else(|e| {
                    panic!("invalid {MAX_DEVICE_ALLOCATION_ENV}={raw:?}: {e}")
                });
                Self::default().with_max_device_allocation(bytes)
            }
            Err(_) => Self::default(),
        }
    }
}

/// Parse a Kubernetes-style byte size: `1024`, `512Mi`, `32Gi`, `32GiB`, `2G`, `2GB`, etc.
///
/// Rules:
/// - No suffix → raw bytes
/// - `K`/`M`/`G`/`T` (optionally with trailing `B`) → decimal (10^3, 10^6, 10^9, 10^12)
/// - `Ki`/`Mi`/`Gi`/`Ti` (optionally with trailing `B`) → binary (2^10, 2^20, 2^30, 2^40)
/// - Whitespace and case are ignored.
pub fn parse_byte_size(input: &str) -> Result<usize, String> {
    let trimmed: String = input.trim().chars().filter(|c| !c.is_whitespace()).collect();
    if trimmed.is_empty() {
        return Err("empty value".into());
    }

    let split_at = trimmed
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(trimmed.len());
    let (num_part, unit_part) = trimmed.split_at(split_at);
    if num_part.is_empty() {
        return Err(format!("no numeric prefix in {input:?}"));
    }
    let value: f64 = num_part
        .parse()
        .map_err(|e| format!("bad number {num_part:?}: {e}"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("value must be finite and non-negative: {input:?}"));
    }

    let multiplier: u64 = match unit_part.to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" => 1_000,
        "m" | "mb" => 1_000_000,
        "g" | "gb" => 1_000_000_000,
        "t" | "tb" => 1_000_000_000_000,
        "ki" | "kib" => 1 << 10,
        "mi" | "mib" => 1 << 20,
        "gi" | "gib" => 1 << 30,
        "ti" | "tib" => 1 << 40,
        other => return Err(format!("unknown size suffix {other:?}")),
    };

    let bytes = value * multiplier as f64;
    if bytes > usize::MAX as f64 {
        return Err(format!("value overflows usize: {input:?}"));
    }
    Ok(bytes as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_raw_bytes() {
        assert_eq!(parse_byte_size("1024").unwrap(), 1024);
        assert_eq!(parse_byte_size("0").unwrap(), 0);
    }

    #[test]
    fn parses_decimal_suffixes() {
        assert_eq!(parse_byte_size("1K").unwrap(), 1_000);
        assert_eq!(parse_byte_size("2MB").unwrap(), 2_000_000);
        assert_eq!(parse_byte_size("3 g").unwrap(), 3_000_000_000);
        assert_eq!(parse_byte_size("4tb").unwrap(), 4_000_000_000_000);
    }

    #[test]
    fn parses_binary_suffixes() {
        assert_eq!(parse_byte_size("1Ki").unwrap(), 1024);
        assert_eq!(parse_byte_size("2MiB").unwrap(), 2 << 20);
        assert_eq!(parse_byte_size("32Gi").unwrap(), 32usize << 30);
        assert_eq!(parse_byte_size("20 GiB").unwrap(), 20usize << 30);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_byte_size("").is_err());
        assert!(parse_byte_size("GiB").is_err());
        assert!(parse_byte_size("12XB").is_err());
        assert!(parse_byte_size("-1").is_err());
    }
}
