/*
 * STUB: gpt_oss_router_gemm requires PTX cp.async.bulk.tensor (TMA) accepted
 * only by CUDA 12.6+ ptxas. We build against CUDA 12.4. Qwen3 dual-model
 * serving doesn't call GPT-OSS routing, so keep the public symbol (for
 * linking) but throw at runtime if anyone hits it.
 */
#include <torch/all.h>

void gpt_oss_router_gemm(torch::Tensor& output, torch::Tensor input,
                         torch::Tensor weight, torch::Tensor bias) {
  TORCH_CHECK(false,
              "gpt_oss_router_gemm is stubbed in this build (requires CUDA "
              ">= 12.6 ptxas for cp.async.bulk.tensor; built against 12.4).");
}
