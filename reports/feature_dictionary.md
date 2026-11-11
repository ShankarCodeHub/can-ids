# Feature dictionary (feature version 1.0)

All features are produced by one implementation,
`canids.preprocessing.features.StreamingFeatureExtractor`. Offline datasets and the
real-time pipeline both feed frames through it one at a time, so training and live
inference compute every feature the same way.

**Conventions**
- **Causal only.** Features use the current frame and earlier frames of the same capture, never later ones.
- **Sentinel for "not available yet": `-1`.** It is used, for example, for the first frame of an ID. Every real value of those features is at least 0, so `-1` can't be confused with a real value.
- **Per-ID statistics are fitted on the TRAINING split's Normal frames only.** These are `is_known_id` and the normal period behind `iat_id_z`. The fitted values are saved in `models/<dataset>/feature_artifacts/per_id_stats.json`.
- **Default windows** (from `configs/data.yaml`): the count window holds the last **N = 100** frames, and the time window covers the last **T = 50 ms**.
- **Not features:** `timestamp`, `label`, `attack_type`, `source_file` and `split` are kept in the processed files for analysis only. Models must use the 25 columns listed below. Raw time and file identity would leak information about the label.

Notation: frame *i* has timestamp *tᵢ*, identifier *idᵢ*, *DLC* data bytes *b₀…b₇*, and global inter-arrival time *Δᵢ = tᵢ − tᵢ₋₁*.
The window *W* contains the last *k ≤ N* frames, and *cₓ* is how many frames in *W* have ID *x*.

## Frame features (identity and payload)

| Feature | Formula / definition | Why it helps | Attacks targeted |
|---|---|---|---|
| `can_id` | arbitration ID as an integer | Some IDs are attack-specific, such as `0x000` in DoS or the spoofed target ID. Vehicle-specific, so not transferable across datasets. | DoS, Spoofing |
| `dlc` | number of data bytes, 0–8 | Injected frames often use an atypical length. ROAD stores every frame as 8 bytes, so this feature carries no information there. | Fuzzy, DoS |
| `b0` … `b7` | payload bytes 0–255; `-1` if the byte is beyond the DLC | Spoofed signal values, all-zero DoS payloads, and random fuzzing bytes | Spoofing, DoS, Fuzzy |
| `payload_entropy` | Shannon entropy of the byte values in *b₀…b_{DLC−1}*: −Σ p·log₂p, range 0–3 bits | Random fuzzing payloads have high entropy. Constant DoS payloads (all `00`) or `FF…FF` payloads have zero. | Fuzzy, DoS |
| `hamming_prev_id` | number of bits that differ between this payload and the previous payload of the same ID (popcount of the XOR; missing bytes count as `0x00`); `-1` for the first frame of an ID | Genuine signals change a few bits at a time. Injected values jump, and masquerade frames alternate with genuine ones. | Spoofing, Fuzzy |
| `is_known_id` | 1 if the ID was seen in Normal training traffic, else 0 | Fuzzing injects IDs the vehicle never uses. | Fuzzy, DoS |

## Timing features

| Feature | Formula / definition | Why it helps | Attacks targeted |
|---|---|---|---|
| `iat_global` | *Δᵢ = tᵢ − tᵢ₋₁* (seconds); `-1` for the first frame of a capture | Flooding shrinks the gap between frames. | DoS, Fuzzy |
| `iat_id` | *tᵢ − t_prev(same ID)* (seconds); `-1` for the first frame of an ID | CAN IDs are periodic. Injecting extra frames of an ID roughly halves its period. | Spoofing |
| `iat_id_z` | *(iat_id − μ_id) / max(σ_id, 0.05·μ_id, 10⁻⁴)*, clipped to ±100. μ_id and σ_id are the ID's normal period mean and standard deviation from TRAIN. 0 if no history or no period is known. | Normalises the gap against that ID's own normal period, so a 10 ms ID and a 1 s ID can be compared. The floor stops near-perfectly periodic IDs from producing huge z-scores. | Spoofing, DoS |

## Rolling-window features (last N = 100 frames, unless stated)

| Feature | Formula / definition | Why it helps | Attacks targeted |
|---|---|---|---|
| `win_rate_fps` | *(k − 1) / (t_newest − t_oldest)* in *W*, capped at 10⁶ | Bus load rises during flooding and injection. | DoS, Fuzzy |
| `win_unique_ids` | number of distinct IDs in *W* | Fuzzing increases ID diversity, and DoS collapses it. | Fuzzy, DoS |
| `win_top_id_share` | maxₓ *cₓ / k* | One ID dominating the window is the DoS signature. | DoS |
| `win_id0_share` | *c₀ₓ₀₀₀ / k* | Share of the highest-priority ID `0x000`, which the HCRL DoS attack floods. | DoS |
| `win_cur_id_share` | *c_idᵢ / k* | Injection raises the share of the targeted ID above its normal level. | Spoofing |
| `win_iat_mean` | mean of *Δ* over frames in *W*; `-1` if none | Average spacing between frames. | DoS, Fuzzy |
| `win_iat_std` | population standard deviation of *Δ* over *W*; `-1` if none | Injection makes the normally regular frame spacing irregular. | Fuzzy, Spoofing |
| `win_id_entropy` | −Σₓ (cₓ/k)·log₂(cₓ/k) | Distribution of IDs in the window: low under DoS, high under fuzzing. | DoS, Fuzzy |
| `win_t_count` | number of frames with *tᵢ − t ≤ 50 ms* (time window) | Rate measured over a fixed duration, so it doesn't depend on N. | DoS, Fuzzy |

## Feature groups for the Phase 7 ablation

| Group | Features |
|---|---|
| identity | `can_id`, `dlc`, `is_known_id` |
| payload | `b0`–`b7`, `payload_entropy`, `hamming_prev_id` |
| timing | `iat_global`, `iat_id`, `iat_id_z` |
| window | the 9 `win_*` features |

## Implementation notes (for the viva)

- **Window statistics are updated in O(1) per frame.** The extractor keeps running counts per ID, a histogram of those counts (for the maximum), the running sum Σ c·log₂c (for entropy, which then equals log₂k − Σc·log₂c / k), and running sums of Δ and Δ² (for the mean and standard deviation). A unit test checks every window feature against a brute-force recomputation on 600 random frames.
- **Floating-point drift.** The running sums of Δ are recomputed exactly every 100,000 evictions, so rounding error can't build up over millions of frames.
- **Per-ID periods are computed between consecutive Normal frames** of the same ID within each training file. Injected frames are excluded, so they can't distort the learned normal period.
- **Cross-dataset caveat.** `can_id`, the payload bytes and `is_known_id` depend on the vehicle. Timing and window features are more transferable. Phase 7 uses this split to choose the common feature subset.
