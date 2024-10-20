//! Safe wrapper around `llama_sampler`.

use std::fmt::{Debug, Formatter};
use std::ptr::NonNull;

use crate::context::LlamaContext;
// use crate::timing::LlamaTimings;
use crate::token::LlamaToken;

pub mod params;

/// Safe wrapper around `llama_sampler`.
#[allow(clippy::module_name_repetitions)]
pub struct LlamaSampler {
    pub(crate) sampler: NonNull<llama_cpp_sys_2::llama_sampler>,
}

impl Debug for LlamaSampler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaSampler")
            .field("sampler", &self.sampler)
            .finish()
    }
}

impl LlamaSampler {
    pub fn new(sampler_chain_params: params::LlamaSamplerChainParams) -> Self {
        let sampler = unsafe {
            NonNull::new(llama_cpp_sys_2::llama_sampler_chain_init(
                sampler_chain_params.sampler_chain_params,
            ))
            .expect("llama_sampler_chain_init returned null")
        };
        Self { sampler }
    }

    /// Initialize a distribution sampler with the given seed and add it to the sampler chain.
    pub fn add_dist(self, seed: u32) -> Self {
        unsafe {
            let dist_sampler = NonNull::new(llama_cpp_sys_2::llama_sampler_init_dist(seed))
                .expect("llama_sampler_chain_init_dist returned null");
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), dist_sampler.as_ptr());
        }
        self
    }

    /// Initialize a top-k sampler with the given top-k value and add it to the sampler chain.
    pub fn add_top_k(self, top_k: i32) -> Self {
        unsafe {
            let top_k_sampler = NonNull::new(llama_cpp_sys_2::llama_sampler_init_top_k(top_k))
                .expect("llama_sampler_chain_init_top_k returned null");
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), top_k_sampler.as_ptr());
        }
        self
    }

    /// Initialize a nucleus sampler with the given top-p & min-keep values and add it to the sampler chain.
    pub fn add_top_p(self, top_p: f32, min_keep: usize) -> Self {
        unsafe {
            let top_p_sampler =
                NonNull::new(llama_cpp_sys_2::llama_sampler_init_top_p(top_p, min_keep))
                    .expect("llama_sampler_chain_init_top_k returned null");
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), top_p_sampler.as_ptr());
        }
        self
    }

    /// Initialize a min-p sampler with the given value and add it to the sampler chain.
    pub fn add_min_p(self, min_p: f32, min_keep: usize) -> Self {
        unsafe {
            let min_p_sampler =
                NonNull::new(llama_cpp_sys_2::llama_sampler_init_min_p(min_p, min_keep))
                    .expect("llama_sampler_chain_init_top_k returned null");
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), min_p_sampler.as_ptr());
        }
        self
    }

    /// Initialize a temperature sampler with the given temp value and add it to the sampler chain.
    pub fn add_temp(self, temp: f32) -> Self {
        unsafe {
            let temp_sampler = NonNull::new(llama_cpp_sys_2::llama_sampler_init_temp(temp))
                .expect("llama_sampler_chain_init_temp returned null");
            llama_cpp_sys_2::llama_sampler_chain_add(self.sampler.as_ptr(), temp_sampler.as_ptr());
        }
        self
    }

    /// Initialize a softmax sampler and add it to the sampler chain.
    pub fn add_softmax(self) -> Self {
        unsafe {
            let softmax_sampler = NonNull::new(llama_cpp_sys_2::llama_sampler_init_softmax())
                .expect("llama_sampler_chain_init_softmax returned null");
            llama_cpp_sys_2::llama_sampler_chain_add(
                self.sampler.as_ptr(),
                softmax_sampler.as_ptr(),
            );
        }
        self
    }

    /// Initialize a repetition penalty sampler and add it to the sampler chain.
    pub fn add_penalties(
        self,
        n_vocab: i32,
        special_eos_id: i32,
        linefeed_id: i32,
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_presence: f32,
        penalize_nl: bool,
        ignore_eos: bool,
    ) -> Self {
        unsafe {
            let penalties_sampler = NonNull::new(llama_cpp_sys_2::llama_sampler_init_penalties(
                n_vocab,
                special_eos_id,
                linefeed_id,
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_presence,
                penalize_nl,
                ignore_eos,
            ))
            .expect("llama_sampler_chain_init_penalties returned null");
            llama_cpp_sys_2::llama_sampler_chain_add(
                self.sampler.as_ptr(),
                penalties_sampler.as_ptr(),
            );
        }
        self
    }

    /// Initialize a tail-free sampler with the given z value and add it to the sampler chain.
    ///
    /// Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
    pub fn add_tail_free(self, z: f32, min_keep: usize) -> Self {
        unsafe {
            let tail_free_sampler =
                NonNull::new(llama_cpp_sys_2::llama_sampler_init_tail_free(z, min_keep))
                    .expect("llama_sampler_chain_init_tail_free returned null");
            llama_cpp_sys_2::llama_sampler_chain_add(
                self.sampler.as_ptr(),
                tail_free_sampler.as_ptr(),
            );
        }
        self
    }

    /// Initialize a typical-p sampler with the given value and add it to the sampler chain.
    pub fn add_typical_p(self, p: f32, min_keep: usize) -> Self {
        unsafe {
            let typical_p_sampler =
                NonNull::new(llama_cpp_sys_2::llama_sampler_init_typical(p, min_keep))
                    .expect("llama_sampler_chain_init_typical_p returned null");
            llama_cpp_sys_2::llama_sampler_chain_add(
                self.sampler.as_ptr(),
                typical_p_sampler.as_ptr(),
            );
        }
        self
    }

    /// Initialize a greedy sampler and add it to the sampler chain.
    pub fn add_greedy(self) -> Self {
        unsafe {
            let greedy_sampler = NonNull::new(llama_cpp_sys_2::llama_sampler_init_greedy())
                .expect("llama_sampler_chain_init_greedy returned null");
            llama_cpp_sys_2::llama_sampler_chain_add(
                self.sampler.as_ptr(),
                greedy_sampler.as_ptr(),
            );
        }
        self
    }

    /// Initialize a dynamic temperature (aka entropy) sampler with the given temp, delta, and exponent,
    /// and add it to the sampler chain.
    pub fn add_dynatemp(self, temp: f32, delta: f32, exponent: f32) -> Self {
        unsafe {
            let dynatemp_sampler = NonNull::new(llama_cpp_sys_2::llama_sampler_init_temp_ext(
                temp, delta, exponent,
            ))
            .expect("llama_sampler_chain_init_temp_ext returned null");
            llama_cpp_sys_2::llama_sampler_chain_add(
                self.sampler.as_ptr(),
                dynatemp_sampler.as_ptr(),
            );
        }
        self
    }

    /// Initialize a mirostat v2 sampler with the given values and add it to the sampler chain.
    pub fn add_mirostat_v2(self, seed: u32, tau: f32, eta: f32) -> Self {
        unsafe {
            let mirostat_sampler = NonNull::new(llama_cpp_sys_2::llama_sampler_init_mirostat_v2(
                seed, tau, eta,
            ))
            .expect("llama_sampler_chain_init_mirostat_v2 returned null");
            llama_cpp_sys_2::llama_sampler_chain_add(
                self.sampler.as_ptr(),
                mirostat_sampler.as_ptr(),
            );
        }
        self
    }

    /// Initialize a mirostat sampler with the given values and add it to the sampler chain.
    pub fn add_mirostat(self, n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> Self {
        unsafe {
            let mirostat_sampler = NonNull::new(llama_cpp_sys_2::llama_sampler_init_mirostat(
                n_vocab, seed, tau, eta, m,
            ))
            .expect("llama_sampler_chain_init_mirostat returned null");
            llama_cpp_sys_2::llama_sampler_chain_add(
                self.sampler.as_ptr(),
                mirostat_sampler.as_ptr(),
            );
        }
        self
    }

    /// Reset the sampler chain.
    pub fn reset(&self) {
        unsafe {
            llama_cpp_sys_2::llama_sampler_reset(self.sampler.as_ptr());
        }
    }

    /// Sample a token from the sampler chain.
    pub fn sample(&self, ctx: &mut LlamaContext, idx: Option<u32>) -> LlamaToken {
        let idx = idx.map(|i| i32::try_from(i).unwrap_or(-1)).unwrap_or(-1);
        let token = unsafe {
            llama_cpp_sys_2::llama_sampler_sample(self.sampler.as_ptr(), ctx.context.as_ptr(), idx)
        };
        LlamaToken(token)
    }

    /// Accept a sampled token.
    pub fn accept(&self, token: LlamaToken) {
        unsafe {
            llama_cpp_sys_2::llama_sampler_accept(self.sampler.as_ptr(), token.0);
        }
    }

    /// Reset the timings for the sampler.
    pub fn reset_timings(&self) {
        unsafe {
            llama_cpp_sys_2::llama_perf_sampler_reset(self.sampler.as_ptr());
        }
    }

    // /// Returns the timings for the sampler.
    // pub fn timings(&mut self) -> LlamaSampleTimings {
    //     let timings = unsafe { llama_cpp_sys_2::llama_perf_sampler(self.sampler.as_ptr()) };
    //     LlamaSampleTimings { timings }
    // }
}

impl Drop for LlamaSampler {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_sampler_free(self.sampler.as_ptr()) }
    }
}
