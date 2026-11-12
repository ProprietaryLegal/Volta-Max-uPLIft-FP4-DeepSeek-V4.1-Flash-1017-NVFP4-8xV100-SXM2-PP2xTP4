
// ===========================================================================
// NVFP4-8R prefill RB launcher (RB 4 patch, 2026-11-12).
// Appended after the unchanged text of skinny_moe_qpn_sm70.cu (sha256
// 31d6ea4f...): the kernel template above is the installed one, byte for byte.
// This is the installed skinny_moe_qpn_sm70's m8n8k4 launcher with its checks
// and grid, restricted to what forward_prefill's launches use (SPLITK 8, MXFP4
// E8M0 scales, m8n8k4 fragments). Where the installed op runs RB 2 (its prefill
// test) this runs RB `rb` (2 or 4); every other launch runs RB 1, as installed.
// ===========================================================================

void nvfp4_8r_qpn_rb_moe_prefill(
    torch::Tensor x, torch::Tensor qcodes, torch::Tensor qscales,
    torch::Tensor gscales, torch::Tensor perm, torch::Tensor gids,
    torch::Tensor goff, int64_t topk, torch::Tensor y_slots, bool x_slot_major,
    int64_t num_tokens, int64_t splitk, int64_t scale_mode,
    int64_t fragment_order, int64_t rb) {
  TORCH_CHECK(fragment_order == ORDER_M8N8K4,
              "nvfp4_8r_qpn_rb: m8n8k4 fragments only");
  TORCH_CHECK(splitk == 8, "nvfp4_8r_qpn_rb: SPLITK 8 only");
  TORCH_CHECK(scale_mode == SCALE_MXFP4_E8M0_32,
              "nvfp4_8r_qpn_rb: MXFP4 E8M0 only");
  TORCH_CHECK(rb == 2 || rb == 4, "nvfp4_8r_qpn_rb: rb in {2, 4}");
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  const c10::cuda::CUDAGuard device_guard(x.device());
  TORCH_CHECK(x.dim() == 2 && y_slots.dim() == 2,
              "x and output must be matrices");
  for (const auto& tensor :
       {qcodes, qscales, gscales, perm, gids, goff, y_slots}) {
    TORCH_CHECK(tensor.device() == x.device(),
                "all skinny MoE tensors must share a device");
  }
  TORCH_CHECK(num_tokens > 0 && topk > 0 && topk <= 65535,
              "positive token count and supported routing width are required");
  TORCH_CHECK(num_tokens <= 65535 / topk, "routing slots exceed CUDA grid-y");
  TORCH_CHECK(gscales.dim() == 1 && gscales.numel() > 0,
              "expert scales must be nonempty");
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(qscales.is_cuda() && qscales.dtype() == torch::kUInt8 &&
              qscales.is_contiguous());
  TORCH_CHECK(gscales.is_cuda() && gscales.dtype() == torch::kFloat &&
              gscales.is_contiguous());
  TORCH_CHECK(perm.is_cuda() && perm.dtype() == torch::kInt &&
              perm.is_contiguous());
  TORCH_CHECK(gids.is_cuda() && gids.dtype() == torch::kInt &&
              gids.is_contiguous());
  TORCH_CHECK(goff.is_cuda() && goff.dtype() == torch::kInt &&
              goff.is_contiguous());
  TORCH_CHECK(y_slots.is_cuda() && y_slots.dtype() == torch::kHalf &&
              y_slots.is_contiguous());
  const int64_t T = num_tokens, K = x.size(1);
  const int64_t E = gscales.size(0), N = y_slots.size(1);
  const int64_t S = T * topk;
  TORCH_CHECK(x.size(0) == (x_slot_major ? S : T), "x rows mismatch");
  TORCH_CHECK(qcodes.numel() == E * N * (K >> 1), "qpn codes size");
  const int64_t sh = 4 + scale_mode;
  TORCH_CHECK(qscales.numel() == E * N * (K >> sh), "qpn scales size");
  TORCH_CHECK(gids.size(0) == S && goff.size(0) == S + 1,
              "compact routing size");
  TORCH_CHECK(perm.size(0) == S && y_slots.size(0) == S);
  TORCH_CHECK(S <= 65535, "grouped qpn MoE: tokens * topk = ", S,
              " exceeds the CUDA grid y limit");
  TORCH_CHECK(K % 64 == 0 && (K / 16) % splitk == 0,
              "K/16 must split into splitk equal slices");
  constexpr int kColumnTiles = 2;
  TORCH_CHECK(N % (32 * kColumnTiles) == 0, "skinny MoE needs N % 64");
  const dim3 grid((unsigned)(N / (32 * kColumnTiles)),
                  (unsigned)std::min(S, E));
  auto stream = at::cuda::getCurrentCUDAStream();
  // The installed op's prefill test, unchanged.
  const bool prefill = (S + E - 1) / E > 8;
#define NVFP4_8R_QPN_RB_LAUNCH(RBv)                                         \
  skinny_nvfp4_moe_qpn<8, SCALE_MXFP4_E8M0_32, RBv, kColumnTiles>           \
      <<<grid, dim3(32 * 8), 0, stream>>>(                                  \
          qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),          \
          gscales.data_ptr<float>(),                                        \
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),            \
          reinterpret_cast<half*>(y_slots.data_ptr<at::Half>()),            \
          perm.data_ptr<int>(), gids.data_ptr<int>(), goff.data_ptr<int>(), \
          (int)N, (int)K, (int)topk, x_slot_major ? 1 : 0)
  if (!prefill)
    NVFP4_8R_QPN_RB_LAUNCH(1);
  else if (rb == 2)
    NVFP4_8R_QPN_RB_LAUNCH(2);
  else
    NVFP4_8R_QPN_RB_LAUNCH(4);
#undef NVFP4_8R_QPN_RB_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(nvfp4_8r_qpn_rb, m) {
  m.def(
      "moe_prefill(Tensor x, Tensor qcodes, Tensor qscales, Tensor gscales, "
      "Tensor perm, Tensor gids, Tensor goff, int topk, Tensor(a!) y_slots, "
      "bool x_slot_major, int num_tokens, int splitk, int scale_mode, "
      "int fragment_order, int rb) -> ()");
}

TORCH_LIBRARY_IMPL(nvfp4_8r_qpn_rb, CUDA, m) {
  m.impl("moe_prefill", &nvfp4_8r_qpn_rb_moe_prefill);
}
