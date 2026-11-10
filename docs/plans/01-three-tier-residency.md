# 01. Three-tier residency: native hot, IQ warm, native host

> Public copy of a build record written on 2026-11-09. Host names, private paths, names and figures from our private traffic were replaced or removed (see `.publication/SOURCE.json`). "Production" means the earlier Volta sPLIt build. `$LANE` is this build's root and `$PROD` the Volta sPLIt checkout. Paths under `logs/` name private receipts that are not published; `research/kernels/` and `research/volta/` are private copies of public sources (links in [docs/README.md](../README.md)).

Read `README.md` (shared rules, measuring method, glossary) and `00-WHERE-TIME-GOES.md` first.
Paths are relative to `$LANE/` unless absolute. Line numbers are of `engine/src` on
2026-11-09 after patches P7 and P8; re-find by function name if they moved.

## 1. Goal and expected gain

Today (two tiers) a routed expert that is not in HBM is read over PCIe inside the decode graph. That
costs 1.0 to 1.4 ms per expert per layer (MEASURED, the test host, `logs/testhost/p3-bench-20261109T185349Z.json`).
This plan removes PCIe reads from the decode path. Each expert of a stage gets one tier:

| Tier | Record | Where | Decode path | Prefill path |
|---|---|---|---|---|
| HOT | native MXFP4, 4,874,240 B per rank | HBM | QPN kernel | QPN kernel |
| WARM | production IQ record, 1.94 to 2.56 MB per rank | HBM (plus native copy in host RAM) | production IQ op | QPN kernel, native copy staged from host RAM |
| HOST | native MXFP4 | host RAM only | QPN kernel reading host RAM (zero-copy) | QPN kernel, staged from host RAM |

Expected decode round time, from parts:

| Part | ms/round | Label and source |
|---|---|---|
| All-native, all in HBM | 36.1 | ESTIMATED (owner; LANE-RESEARCH §2) |
| IQ op called once per layer on the warm routes | 0 to 7.4 | Upper bound MEASURED (owner): production pays 7.4 ms with all 36 routes live. Cost at few live routes is not measured; §8 measures it |
| Tier mask, FP32 add, id lookups (3 small ops per layer) | at most 1.5 | ESTIMATED; each is smaller than the whole two-tier split, which adds 63.5 to 101.4 us per layer (MEASURED) |
| HOST launches kept in the graph | 4.06, or 0 with residency `hot_iq` | MEASURED on the test host for the two-tier split |
| Total | 36.1 to 49.1 | ESTIMATED |

At 3.12 tokens per round (114.4 tok/s x 27.3 ms, MEASURED owner) that is 63.6 to 86.5 tok/s. The
two-tier estimate is 96.9 ms/round (ESTIMATED at the old budgets); the measured 8-card baseline is in
`00-WHERE-TIME-GOES.md`. Expected gain: the baseline minus 36.1 to 49.1 ms.

Cost: warm routes run at IQ precision in decode. LANE-RESEARCH §3 put them at 6.68 % (stage 0) and
12.58 % (stage 1) of decode routes for hot sets of 3,430 and 2,282 experts. With the HBM measured on
the 8-card boot the hot sets are larger (§5.3), so the IQ share is lower. Step 9 logs the real shares.

## 2. Why it should work; what would falsify it

- All 384 IQ experts per layer fit beside the dense weights today: production does it. IQ expert
  bytes per card are 14.72 GiB (stage 0) and 16.03 GiB (stage 1) (MEASURED from the GGUF headers,
  2026-11-09). Free HBM before experts is 29.30 / 28.86 GiB (MEASURED, `run/state/p6-boot2.log`).
- The production IQ op drops a route whose id is negative (comment "id < 0: route dropped" in
  `engine/src/benchmarks/deepseek_v41/v41k/v41k.cu` `moe_plan` and in an internal v41k variant
  `plan_warp`). So one call with ids = warm slot or -1 computes only the warm routes.
- Native and IQ outputs are both additive partials of the same FP sum, so they can be added per rank
  before the TP all-reduce (§5.6).

Falsifiers. Stop and report if one holds:
1. The deployed `v41k.so` does not drop id < 0 routes, or leaves unwritten rows (check A3 case c4).
   The deployed library has no matching source in `engine/src`, so this must be run, not read.
2. The IQ op at few live routes costs close to its full 185 us per layer, and the HOST launches
   another 100 us: then the round is near 49 ms (64 tok/s), below the operator's 80 tok/s bar.
3. Quality: three-tier output is no closer to all-native than production is (§8 rule).

## 3. Upstream-overlap check

| Row | Verdict |
|---|---|
| §0.9 route 1 (mixed pack) | Adopted. This plan is route 1 |
| §0.9 route 3 (host-RAM offload) | Partly: kept only for the HOST tier and for prefill |
| §0.9 route 4 (12 GPUs) | Not adopted: ruled out by the operator (8 cards) |
| §0.5 v100-skinny QPN | Adopted unchanged: native tiers use `skinny_moe_qpn_sm70` |
| §0.6 PR #1277 (offload 0.33, 27 misses per round, 53.5 ms/round) | Not adopted as a path; it is the evidence that misses dominate |
| §0.4 Volta roofline (FP16 tensor cores only) | Respected: no new kernel |
| §0.7 traps | Read before step 6 |

No duplicate: no upstream PR mixes native and IQ experts per expert id in one layer.

## 4. Code map

`engine/src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_native.py` (620 lines):

| Lines | Item | Use |
|---|---|---|
| 41-54 | constants, `EXPERT_BYTES == 4_874_240` | reuse |
| 111-214 | `NativeExpertSet`; `from_raw` 162-173 (HBM), `from_raw_host` 175-210 (registered host RAM) | reuse unchanged |
| 220-259 | `PrefillStaging`, `prefill_staging` (one per card, patch P8) | reuse unchanged |
| 262-273 | `split_maps` | reuse |
| 276-500 | `NativeV41Moe`: `__init__` 288-323, `_compact` 325-341, `_groups` 343-346, `_split_groups` 348-363, `forward` 365-394, `forward_prefill` 403-472, `_run_cold` 474-500 | edit `__init__`, `forward`; add `_tier_groups` |
| 503-530 | `choose_residency` | keep for `two_tier` |
| 533-573 | `EXTRA_KEYS`, `TALLY_PHASE`, `NativePackConfig` (`from_extra` 550-568, `decode_counts` 570-573) | edit |
| 581-603 | `StagePlan`, `plan_stage` | edit |
| 606-620 | `build_layer` | edit |

Other files:

| File (under `engine/src/vllm/`) | Lines | Item |
|---|---|---|
| `model_executor/model_loader/gguf_loader.py` | 54-75 | `__init__`: allowed extra keys |
| same | 76-87 | `drop_routed_experts` |
| same | 89-141 | `_attach_nvfp4_8r` (per-rank free HBM, plan, build, log line at 129) |
| same | 774-778, 811-812 | `load_model`: drop of routed experts; attach call |
| `model_executor/layers/quantization/gguf_moe.py` | 452-469 | `_apply_nvfp4_8r`: M 1..6 -> `forward`, else `forward_prefill` |
| `model_executor/layers/quantization/gguf_v41_moe.py` | 136-153 | `_expert_major_ready` |
| same | 205-220 | `ROWS_BYTES`, `_PEER`, `register_rows_peer(device, rank, pointers, own)` |
| same | 263-334 | `_v41_experts_rows_fast` (the IQ decode path used here) |
| same | 538-694, 703-710 | `_v41_experts`; op `torch.ops.vllm.gguf_v41_original_expert_dp4a` |
| `model_executor/layers/quantization/gguf_native.py` | 91-105 | `empty_guarded_weight(shape, weight_type, device)` |
| `model_executor/layers/quantization/gguf_expert_storage.py` | 134-187 | `_source(tensor)`, `_read(fd, ranges, out)` |
| `model_executor/layers/quantization/gguf_shared_layout.py` | 21, 110-233 | `DOWN_ROWS`; `SharedExpertLayout.for_type(...)`, `.pack(original)` |
| `model_executor/layers/quantization/gguf_shared_prefill.py` | 212-221, 330-374 | `_DeviceView`; `_init_rows` (how production registers the peer bank) |
| `distributed/device_communicators/custom_all_reduce.py` | 1014-1036 | `CustomAllreduce.create_shared_buffer(size, group)` (collective) |
| `model_executor/layers/fused_moe/sm70/reduction.py` | | `_mtp_weighted_reduce(expert_output, topk_weights, output)` |

Tests to copy patterns from, in `engine/src/benchmarks/nvfp4_8r/`: `accept_p2.py` (`Inconclusive`,
`gpu_gate`, `rel_l2`, receipt JSON), `accept_p3_layer.py` (`graph_of`), `make_p2_refs.py`
(`contribution`, fp64 reference from raw NVFP4), `test_p3_split_groups.py` (kernel grid rule
emulated on CPU), `test_residency.py`, `test_p3_loader_wiring.py`.

Facts from the code that the design depends on:
- The QPN kernel skips a group with `goff[g+1] <= goff[g]`; its grid covers min(slots, bank) groups.
- The IQ op needs `gate.shape[1] == 576`, `down.shape[1] == 1280`, M 1..6, types in (16, 17, 29).
- Its result is FP16 [M, 5120]: this rank's 1,280 output rows hold the weighted sum over live
  routes; the other rows are zeros. Router weights are applied inside the op.
- Every rank must call the IQ op once per layer per step: `peer_gather` waits for the other three.
- GGUF routed-expert tensors differ by layer: types IQ1_M (29, 56 B per 256 weights), IQ2_XXS
  (16, 66 B), IQ2_XS (17, 74 B). Gate and up have the same type in every layer (MEASURED, headers).

## 5. Design

### 5.1 Types (new pure module `nvfp4_8r_tiers.py`, same directory; numpy only, no torch.cuda)

```python
class Tier(enum.IntEnum): HOT = 0; WARM = 1; HOST = 2

@dataclass(frozen=True)
class TwoTier: pass                      # today's build: hot in HBM, the rest zero-copy from host
@dataclass(frozen=True)
class HotIq: pass                        # HOT + WARM, no HOST tier
@dataclass(frozen=True)
class HotIqHost: route_share: float      # 0.0 <= route_share <= 0.05
Residency = TwoTier | HotIq | HotIqHost

@dataclass(frozen=True)
class TierPlan:                          # one stage; identical on its four ranks
    layers: range
    tiers: dict[int, np.ndarray]         # layer -> int8 [384] of Tier values
    budget_bytes: int
    hbm_bytes: int                       # bytes the plan uses per card
    digest: str                          # sha256 of the tier arrays in layer order
```

Invariants, checked in `TierPlan.__post_init__` (raise on any breach):
- every array is int8 [384] with values in {0, 1, 2}; `hbm_bytes <= budget_bytes`;
- every layer has at least one HOT and at least one WARM expert (a layer without either is refused);
- `HotIq` plans contain no HOST.

Derived per layer on the device (int32 [384], -1 = absent): `slot_of_expert` (>= 0 iff HOT),
`cold_slot_of_expert` (>= 0 iff not HOT; the host-RAM native set holds WARM and HOST experts),
`host_slot_of_expert` (= cold slot iff HOST), `iq_slot_of_expert` (>= 0 iff WARM). Slots are in
ascending expert id. An expert is never in two decode tiers: decode reads the single `Tier` array.

```python
@dataclass(frozen=True)
class IqWarmBank:                        # in nvfp4_8r_native.py; one layer, one rank, HBM
    w1: torch.Tensor                     # uint8 [W, 576, 20 * bb13]
    w3: torch.Tensor                     # uint8 [W, 576, 20 * bb13]
    w2: torch.Tensor                     # uint8 [W, 1280, 9 * bb2], zero tail after the bank
    type13: int                          # GGML type of gate and up
    type2: int                           # GGML type of down
```

Owners: the hot set owns its HBM; the cold set owns its registered host arena (`from_raw_host`);
`IqWarmBank` owns its HBM; the peer bank has one owner per card (module-level, §5.5); prefill staging
stays one per card (P8). Nothing is shared with production.

### 5.2 Config (model-loader-extra-config), parsed once in `NativePackConfig.from_extra`

| Key | Value | Rule |
|---|---|---|
| `nvfp4_8r_residency` | `two_tier`, `hot_iq` or `hot_iq_host` | required |
| `nvfp4_8r_host_route_share` | float in [0, 0.05] | required with `hot_iq_host`; an error with the other two |

Add `nvfp4_8r_residency` to `EXTRA_KEYS` (all required together, as today). The share key is
conditional, so it is not in `EXTRA_KEYS`: name it in a second tuple `CONDITIONAL_KEYS` that
`gguf_loader.__init__` also allows. Parse both into `Residency`. First boot of this plan:
`hot_iq_host` with `0.0`. No defaults in code.

Environment required when the residency has a WARM tier (checked at attach; raise if not):
`VLLM_V41_DOWN_ROWS=1`, `VLLM_V41_EM=1`, `VLLM_V41_EXPERT_FAST=1`, `VLLM_V41_PLAN_FUSE=1`,
`VLLM_V41_ROWS_TILE=4,8`, and the v41k library loaded (`_expert_major_ready()` true).
`run/run_serve.sh` already exports them.

### 5.3 Choosing the tiers (pure function)

`choose_tiers(counts, layers, iq_bytes, budget_bytes, residency) -> TierPlan`
- `counts`: the decode tally [40, 384] from `NativePackConfig.decode_counts()` (code reads
  `data["phases"]` and `data["counts"][phase]`; the file is private: never print or save per-expert
  values).
- `iq_bytes[layer] = 11520 * (2 * bb13 + bb2)`, bb from the layer's GGUF types (56, 66 or 74). This
  is 576 rows x 20 blocks for gate and for up, plus 1,280 rows x 9 blocks for down.

Rule:
1. HOST set. `HotIq`: empty. `HotIqHost(s)`: sort the stage's experts ascending by
   (count, layer, expert); take the longest prefix whose summed count <= s x (stage total count).
   With s = 0.0 that is exactly the experts with count 0.
2. Every other expert starts WARM. `need = sum(iq_bytes[layer])` over them. If `need > budget_bytes`
   raise ("budget cannot hold the IQ tier").
3. Promote WARM to HOT. Sort candidates descending by `Fraction(count, EXPERT_BYTES - iq_bytes[layer])`,
   ties ascending (layer, expert). Walk the list; promote while
   `need + (EXPERT_BYTES - iq_bytes[layer]) <= budget_bytes`; stop at the first that does not fit.
4. Build `TierPlan` (its invariants raise).

`budget_bytes = min(free bytes over the stage's four ranks) - reserve_bytes`. Fixed small buffers
(0.62 MB per layer of `y13/inter/y2/out`, the 94,464-byte peer bank, slot maps) stay inside the reserve.

Arithmetic with the measured boot (free 29.30 / 28.86 GiB, reserve 6 GiB; budgets 25.02e9 and
24.55e9 B; mean IQ record 2,057,472 and 2,241,792 B). All ESTIMATED from layer means:

| Stage | All-IQ bytes | Left for promotion | Hot, no HOST tier | Hot, HOST = zero-route (360 / 606) | Native records in host RAM, 4 ranks |
|---|---|---|---|---|---|
| 0 | 15,801,384,960 | 9.22e9 | 3,272 | 3,535 | 80.8 to 85.9 GB |
| 1 | 17,216,962,560 | 7.33e9 | 2,784 | 3,300 | 85.4 to 95.5 GB |

Each GiB of reserve released promotes about 381 (stage 0) or 408 (stage 1) experts (plan 02).

### 5.4 Loading (in `_attach_nvfp4_8r`, all four ranks of a stage in the same order)

1. `free` as today. `dist.all_gather_object` over `get_tp_group().cpu_group`; use the minimum.
2. Types and sources of this stage's routed-expert GGUF tensors. They must be captured in
   `load_model` before line 777 drops them: for each `(raw, name)` with `is_routed_expert_name(name)`,
   not `is_pp_missing_parameter(name, weight_model)`, `prefix, projection, _ = name.rsplit(".", 2)`,
   projection in gate_proj/up_proj/down_proj -> w1/w3/w2, `layer_id = weight_model.get_submodule(prefix).layer_id`
   (`is_pp_missing_parameter` is in `vllm.model_executor.models.utils`; same parsing as
   `gguf_expert_storage.prepare_original_expert_arena`, lines 43-125).
   Keep `self._nvfp4_iq_sources[layer_id][shard] = raw`. The reader tensor is `self._native_tensors[raw]`
   (`.data` memmap [384, rows, row_bytes], `.tensor_type`; rows = 2,304 for w1/w3 and 5,120 for w2).
   Raise unless each layer has all three shards, every type is 16, 17 or 29, and the w1 type equals
   the w3 type.
3. `choose_tiers`. Gather `plan.digest` from all ranks; raise unless the four are equal.
4. Peer bank (§5.5).
5. Per layer: read the pack records as today; `hot = raw[HOT ids]` -> `from_raw`; all non-HOT ->
   `from_raw_host`; `IqWarmBank.from_gguf(...)` for the WARM ids:
   - `fd, offset = _source(tensor)`; raise if `_source` returns None;
   - stride = rows x row_bytes of the tensor. w1/w3: range `(offset + e*stride + rank*576*width, 576*width)`
     per warm expert e. w2: `(offset + e*stride + rank*1280*width, 1280*width)`;
   - `_read(fd, ranges, pinned.numpy())`, then `bank = empty_guarded_weight((W, rows, width), type, device)`,
     `bank.copy_(pinned)`.
6. Log one line per rank: counts of HOT/WARM/HOST, share of tally routes per tier, free after. Counts
   and shares only.

### 5.5 Peer bank

`gate_up_fast` writes each rank's intermediate rows into an IPC bank; `peer_gather` reads the other
ranks' rows. Production registers the bank in `SharedOriginalPrefill._init_rows`, which this build
never constructs. New function `attach_iq_peer(cpu_group, rank, device)` in `nvfp4_8r_native.py`:

```python
pointers = CustomAllreduce.create_shared_buffer(moe.ROWS_BYTES, cpu_group)
own = torch.as_tensor(_DeviceView(pointers[rank], (moe.ROWS_BYTES,), "|u1"), device=device)
moe.register_rows_peer(device, rank, pointers, own)
torch.cuda.synchronize(device); dist.barrier(group=cpu_group)
```

(`moe` = `gguf_v41_moe`; `_DeviceView` from `gguf_shared_prefill`.) `device` must be the indexed
device (`torch.device("cuda", torch.cuda.current_device())`), because the op looks up
`_PEER[x.device]`. After it, raise unless `moe._PEER.get(device)` is set. Call once per card; a second
call for the same card raises.

### 5.6 Decode dispatch, one captured graph (`NativeV41Moe.forward`)

`__init__` gains `host_slot_of_expert`, `iq_slot_of_expert`, `iq: IqWarmBank | None`, and
`host_in_decode: bool` (true only for `HotIqHost`). The launch list is fixed at construction, so the
graph shape never depends on data. `two_tier` objects keep `_groups` / `_split_groups`. New
`_tier_groups(ids)` for objects with a WARM tier:

```python
flat = ids.reshape(-1).long(); n = flat.numel()
nh, nc = self.experts.num_slots, self.cold.num_slots
hot = self.slot_of_expert[flat].long()
host = self.host_slot_of_expert[flat].long()
key = torch.where(hot >= 0, hot, torch.where(host >= 0, nh + host, nh + nc))  # WARM -> sentinel
perm, gids, goff, sorted_e, new_group = self._compact(key)
count_hot = (key < nh).sum(); count_native = (key < nh + nc).sum()
gids_h = torch.where(gids < nh, gids, 0).to(torch.int32)
goff_h = torch.minimum(goff, count_hot).to(torch.int32)
launches = [(self.experts, perm, gids_h, goff_h)]
if self.host_in_decode:
    groups_hot = (new_group & (sorted_e < nh)).sum()
    idx = (torch.arange(n + 1, device=key.device) + groups_hot).clamp(max=n)
    goff_c = torch.minimum(goff[idx], count_native).to(torch.int32)
    gids_c = (gids[idx[:n].clamp(max=n - 1)] - nh).clamp(min=0, max=nc - 1).to(torch.int32)
    launches.append((self.cold, perm, gids_c, goff_c))
return launches
```

Why it is right: keys sort HOT slots, then HOST slots, then the sentinel. Clamping `goff` to
`count_hot` (hot launch) or `count_native` (host launch) makes the sentinel group and every later
group empty, and the kernel skips empty groups. The `gids` clamp keeps a skipped group's id in range.

Then, in `forward`:
1. W13 launches, SwiGLU, W2 launches exactly as today, over `launches`.
2. `iq_ids = self.iq_slot_of_expert[topk_ids.long()]` (int32 [M, 6]; -1 for HOT and HOST routes).
3. `w_native = torch.where(iq_ids < 0, topk_weights, 0.0)`; `_mtp_weighted_reduce(y2, w_native, out)`.
4. `iq_out = torch.ops.vllm.gguf_v41_original_expert_dp4a(x, iq_ids, topk_weights, iq.w1, iq.w3, iq.w2, iq.type13, iq.type2, 0, SWIGLU_LIMIT)`.
5. Return `out + iq_out.float()` (FP32; `_apply_nvfp4_8r` casts to FP16 as today).

Same slot rows: the native kernels never write the slot rows of WARM routes, so those rows of `y13`
and `y2` hold old values. Allocate `y13` and `y2` with `torch.zeros` (not `torch.empty`) so an
unwritten row is finite; step 3 multiplies it by 0.0. Never read `y2` anywhere else.

Why the sum is exact: the layer output is sum over ranks. The native partial of rank r is the K-slice
r of every native route on all 5,120 rows. The IQ partial of rank r is the full-K result of every
warm route on rows r*1280..r*1280+1279. The TP all-reduce adds the four ranks.

### 5.7 Prefill

No change to `forward_prefill`: HOT from HBM; every non-HOT expert the chunk routes to is staged from
the host-RAM native set, 32 at a time. So prefill is all-native, as today, but the cold set is larger
(§5.3 table). Cost: about 207 to 245 non-HOT experts per layer; a large chunk routes to nearly all of
them (ESTIMATED); `_run_cold` gathers 4.87 MB per expert at 4.8 GB/s (MEASURED: 1,014.8 us per expert,
SM gather). That is 0.21 to 0.25 s per layer, 4.2 to 5.0 s per stage per chunk (ESTIMATED). Today's
two-tier build stages 127 to 132 per layer (MEASURED counts). Step 9 measures it; plan 04 cuts it.

### 5.8 CUDA graph capture sizes 1 to 6

Unchanged: `cudagraph_capture_sizes: [1..6]`, `max_tokens=6`. Every shape depends on M only. No
tensor is read on the host (`.sum()` results stay on the device). The IQ op is the same op
production captures at M 1..6. M above 6 goes to `forward_prefill`.

## 6. Build steps

Edit only under this lane's `engine/` and `run/`. Before each edit copy the file to `patches/T3-before/`; after it write
`diff -u` to `patches/T3-NN-WHAT.patch` (one per step). After each Python edit copy the file to
both trees (`engine/src/vllm/...` and `engine/venv/lib/python3.12/site-packages/vllm/...`), run
`cmp` on the two, and prove the import path:
`CUDA_VISIBLE_DEVICES= engine/venv/bin/python -c "import vllm.model_executor.layers.fused_moe.experts.nvfp4_8r_native as m; print(m.__file__)"`
must print the site-packages path. CPU tests run as
`cd engine/src && CUDA_VISIBLE_DEVICES= PYTHONPATH=$LANE/engine/venv/lib/python3.12/site-packages:$LANE/engine/src ../venv/bin/python -m benchmarks.nvfp4_8r.NAME`.

1. Write the four checks of §7 (A1 to A4). Run A1 and A2: they must fail (missing module). Keep the output.
2. `nvfp4_8r_tiers.py`: `Tier`, `Residency`, `TierPlan`, `iq_record_bytes(type13, type2)`,
   `choose_tiers`. Verify: A1 passes.
3. `NativePackConfig`: new keys, `residency` field. `gguf_loader.__init__` (lines 56-69) must allow
   both keys. Add `nvfp4_8r_residency: two_tier` to `run/serve-nvfp4-pp2.yaml` (no share key). Verify:
   `test_p3_loader_wiring` still passes; a profile with `hot_iq_host` and no share raises.
4. `IqWarmBank.from_gguf`. Verify on CPU (device `cpu`) for layers 5, 12, 27, ranks 0 and 3, eight
   experts each: bank bytes equal `SharedExpertLayout.for_type(5120, 2304, 4, rank, role, type).pack(np.asarray(tensor.data[e]))`
   with `VLLM_V41_DOWN_ROWS=1`.
5. `NativeV41Moe`: new fields, `_tier_groups`, `forward` changes, `torch.zeros` buffers. `two_tier`
   keeps `_split_groups` untouched. Verify: A2 passes; `test_p3_split_groups` still passes.
6. `attach_iq_peer`; `build_layer` takes the layer's `Tier` array and optional IQ sources and
   returns the three-tier object. Verify: A3 on the test host.
7. `gguf_loader`: capture `_nvfp4_iq_sources` before the drop; `_attach_nvfp4_8r` per §5.4. For
   `two_tier` the old path runs unchanged. Verify: wiring test on CPU; `cmp` both trees.
8. Boot with `two_tier` on :18043 (the maintainer's window; the host's own stop and restore commands only). Verify:
   same hot/cold counts in the log as before, and `--compare` token-exact against the baseline jsonl.
   This proves the edit did not change the old path.
9. Set `hot_iq_host`, share `0.0`. Boot. Verify the log line, then run A4 and §8.

## 7. Acceptance checks (written first, seen failing)

Receipts are JSON with `status` PASS / FAIL / INCONCLUSIVE under `logs/`. Exit codes 0 / 1 / 3.

A1 `test_t3_tiers.py` (CPU). Expected values by hand arithmetic, written in the test:
- `iq_record_bytes`: (29,29) 1,935,360; (16,16) 2,280,960; (17,17) 2,557,440; (17,29) 2,350,080;
  (16,29) 2,165,760; (29,16) 2,050,560 (11520 x (2 x bb13 + bb2)).
- A 2-layer toy stage with hand-chosen counts and budget: the exact HOST/WARM/HOT sets, worked out
  in comments. Cases: share 0.0; share that admits exactly one non-zero expert; a budget one byte
  short of a promotion; a budget below all-IQ (must raise); a layer left without HOT (must raise).
- Real tally run on the 8-card host: prints only tier counts, `hbm_bytes <= budget`, route share per tier.
  INCONCLUSIVE if the tally file is absent.

A2 `test_t3_dispatch.py` (CPU). For 2,000 random routings at M 1..6 over a random tier map: emulate
the kernel rule (group g covers `perm[goff[g]:goff[g+1]]`, skipped if empty, grid = min(slots, bank))
for each launch. Expected, computed route by route from the tier map without `_compact`: every HOT
route is covered once by the hot launch with its hot slot; every HOST route once by the host launch
with its cold slot; no WARM route by any launch; `iq_ids` equals the warm slot or -1; `w_native` is
0 exactly on WARM routes. Include M where all routes are WARM and where none are.

A3 `accept_t3_layer.py` (the test host, one card, four processes = four ranks, gloo group, real packs of
layers 5, 12, 27 and their IQ slices). It calls the product functions (`build_layer`,
`attach_iq_peer`, `forward`). Inputs x, ids, weights as `make_p2_refs.py` builds them. Reference in
fp64 on CPU: native routes from the raw NVFP4 formula (`make_p2_refs.contribution`), warm routes
from `gguf.quants.dequantize` of the GGUF tensors, same SwiGLU limit. Compare the sum of the four
ranks' outputs with the reference by relative L2:
- c1 every expert the input routes to is HOT, the rest WARM: <= 1e-2 (the P2 rule; P2 measured 4.78e-4).
- c2 every expert the input routes to is WARM, the rest HOT: record e0. Rule <= 2e-2, from Q8 arithmetic: activations are quantized to
  int8 per 32-block twice (step 1/127 of the block maximum). If e0 > 2e-2, exit INCONCLUSIVE: the
  tolerance derivation is wrong; do not loosen it yourself.
- c3 mixed (HOT if id % 4 < 2, WARM if id % 4 == 2, HOST if id % 4 == 3): <= 2e-2.
- c4 tokens with no WARM route: before `forward`, allocate and free a NaN tensor of x's shape; the IQ
  partial must be exactly zero and the output finite.
- c5 CUDA graph per M 1..6: replay with ids A, then ids B (different tier mix); replay B must be
  bit-equal to eager B.
- c6 the native partial of c3 (`out` before the IQ add) is bit-equal to the all-resident object
  (`NativeV41Moe(experts, slot_of_expert, max_tokens)`, today's path) run with the WARM routes' weights set to 0.
INCONCLUSIVE (exit 3): `gpu_gate` refuses, a pack or IQ slice is missing, or the IPC bank cannot be
mapped. Command: a runner `run_t3_layer_testhost.sh CARD` modelled on `run_p3_layer_testhost.sh`.
The IQ slices for the test host come from a CPU script on the 8-card host (`make_t3_iq_slices.py`: per layer and rank,
the three sliced banks of all 384 experts, 0.7 to 1.0 GB per layer-rank).

A4 on the real server at :18043 after step 9: boot log shows four equal digests per stage and no
error; `ab_t1_v2.py --tput` runs twice give equal `toksha` per prompt (self-determinism). FAIL if a
toksha differs between the two runs. INCONCLUSIVE if the server is not the three-tier boot (check
the residency log line).

## 8. Measurement on the real server

Arms: N2 = `two_tier`; T3 = `hot_iq_host` 0.0; T3n = `hot_iq`; P = production's same-day figures
(the maintainer's BASELINE table). One arm = one boot. Per arm: `warmup.py URL`, then
`ab_t1_v2.py --tput http://127.0.0.1:18043 3 logs/t3/tput-v2-LABEL.jsonl | tee logs/t3/ab-LABEL-tput.out`,
then once more with label `LABELr` (within-boot repeat). Boot order N2a, T3a, T3b, N2b, then T3n.
Score with `pairs_arms_v2.py --metric dec --from-out logs/t3 N2a T3a N2b T3b` (read its docstring for
the file names it expects). Noise floor: the larger of the within-boot repeat differences and
|N2a - N2b|. A gain counts if both pairs agree in sign and the mean difference is above twice the
noise floor. Both harness scripts are in `$PROD/1cat-1277/bench5/` (read-only). Prefill:
record `ttft` from the tput lines; the long-prefill measurement is defined in plan 04 §8.

Quality (expected to differ: warm routes are IQ precision). New script `quality_t3.py`: for the three
public prompts of `ab_t1_v2.py` (`TPUT_PROMPTS`), greedy, thinking off, 1,024 tokens, `logprobs` with
`top_logprobs: 5`. Reference = N2. For arm X in (T3, P): first index where the token differs from
N2, and mean absolute difference of the chosen token's logprob over the common prefix. Rule: PASS if
T3's first-divergence index is >= P's on at least two prompts and T3's mean logprob difference is
below P's on at least two. INCONCLUSIVE if the N2 or P receipt is missing or N2 is not
self-deterministic. Receipts: `logs/t3/quality-ARM.json` (token ids and logprobs only; no text).

## 9. Risks, traps, rollback, done

- Rank symmetry: if two ranks of a stage disagree on one tier, `peer_gather` reads rows of a different
  route set and the output is wrong without an error. The digest check is the guard; never skip it.
- A collective call (`all_gather_object`, `create_shared_buffer`, `barrier`) hangs if one rank raised
  earlier. Read all four worker logs before concluding it is a hang.
- 0 x inf: if a native kernel ever writes inf to `y2`, later rounds that mask that slot give NaN.
  A3 c5 would show it. Do not add a NaN filter; find the inf.
- Host RAM: 166 to 181 GB registered (§5.3) against 101 GB today. Check `MemAvailable` before boot.
- Zero-size banks: a layer with no WARM expert is refused by `TierPlan`; do not special-case it.
- Do not reuse `SharedOriginalExpertCache` for the warm bank: its hot set is the first N ids, not a chosen set.
- Lost levers (LANE-RESEARCH §7), not to be re-proposed here: expert top-p prune, async scheduling,
  GATE_COLS/RF, SPINLOOP_EXT, FUSE_TRANSITION, SM70_MOE_ADD_ALLREDUCE env-only, 200 W, L2 warm-up,
  dmv13_xd, K4/K3 drafts, prefetch / evict-first.
- Data boundary: tier arrays and counts stay in memory. Logs and receipts carry counts of experts and
  route shares only.

Rollback: set `nvfp4_8r_residency: two_tier` (same code, old path). Full rollback: restore the files
in `patches/T3-before/` to both trees and `cmp`.

Done when: A1 to A4 PASS with receipts; §8 shows a gain above the noise floor against N2; the
quality rule is PASS; the numbers are in docs/measurements.md; production restored and healthy.
