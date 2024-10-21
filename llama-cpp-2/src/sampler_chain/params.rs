//! A safe wrapper around `llama_sampler_chain_params`.
use std::fmt::Debug;

/// A safe wrapper around `llama_sampler_chain_params`.
///
/// Generally this should be created with [`Default::default()`] and then modified with `with_*` methods.
///
/// # Examples
///
/// ```rust
/// use llama_cpp_2::sampler_chain::params::LlamaSamplerChainParams;
///
///let sparams = LlamaSamplerChainParams::default()
///     .with_no_perf(true);
///
/// assert_eq!(sparams.no_perf(), true);
/// ```
#[derive(Debug, Clone)]
#[allow(missing_docs, clippy::module_name_repetitions)]
pub struct LlamaSamplerChainParams {
    pub(crate) sampler_chain_params: llama_cpp_sys_2::llama_sampler_chain_params,
}

/// SAFETY: we do not currently allow setting or reading the pointers that cause this to not be automatically send or sync.
unsafe impl Send for LlamaSamplerChainParams {}
unsafe impl Sync for LlamaSamplerChainParams {}

impl LlamaSamplerChainParams {
    /// Set no perf to false to enable calculating sampling performance metrics.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_2::sampler_chain::params::LlamaSamplerChainParams;
    /// let params = LlamaSamplerChainParams::default()
    ///     .with_no_perf(false);
    /// assert_eq!(params.no_perf(), false);
    /// ```
    #[must_use]
    pub fn with_no_perf(mut self, no_perf: bool) -> Self {
        self.sampler_chain_params.no_perf = no_perf;
        self
    }

    /// Get the no perf configuration for sampling.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_2::sampler_chain::params::LlamaSamplerChainParams;
    /// let params = LlamaSamplerChainParams::default();
    /// assert_eq!(params.no_perf(), true);
    /// ```
    #[must_use]
    pub fn no_perf(&self) -> bool {
        self.sampler_chain_params.no_perf
    }
}

/// Default parameters for `LlamaSamplerChain`. (as defined in llama.cpp by `llama_sampler_chain_default_params`)
/// ```
/// use llama_cpp_2::sampler_chain::params::LlamaSamplerChainParams;
/// let params = LlamaSamplerChainParams::default();
/// assert_eq!(params.no_perf(), true, "no_perf should be true");
/// ```
impl Default for LlamaSamplerChainParams {
    fn default() -> Self {
        let sampler_chain_params = unsafe { llama_cpp_sys_2::llama_sampler_chain_default_params() };
        Self {
            sampler_chain_params,
        }
    }
}
