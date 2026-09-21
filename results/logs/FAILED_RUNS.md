# Failed and abandoned runs (Rule 9: never delete)

## ced_baseline attempt 1: learned absolute positions (2026-09-18)

- Files: `ced_baseline_attempt1_abspos_seed1.jsonl`, `ced_baseline_attempt1_abspos.log`
- Model: nn.Transformer encoder/decoder, d_model=64, 2 enc + 3 dec layers, learned absolute
  position embeddings, 302k params, lr 6e-4, batch 32, filler curriculum 0 -> 18.
- Result: stopped at step 3500; held-out accuracy flat at ~21% (chance 12.5%), all question types.
- Side diagnostic: same model, direct questions only, no filler, lr 1e-3: ~21% after 1500 steps.
- Diagnosis: even single-fact lookup was not learned, so the failure is in binding entity and value
  tokens within a chunk, not in multi-hop reasoning. Learned absolute positions force the encoder to
  learn that binding separately at every offset.
- Action: switched the baseline to rotary position embeddings (attempt 2).

## ced_baseline attempt 2: RoPE + adaptive curriculum, d_model=64 (2026-09-18)

- Files: `ced_baseline_attempt2_rope_curriculum_seed1.jsonl`, `ced_baseline_attempt2_rope_curriculum.log`
- Model: custom RoPE encoder/decoder, d_model=64, 2 enc + 3 dec layers, lr 1e-3, batch 32.
- Diagnostics before this run: the same model ignored the query entity entirely on a 24-token
  associative-recall probe (identical predictions for any query entity, 6000 steps), but solved the
  probe (100% at 12 pairs) once trained with an entity-count curriculum starting at 2 pairs.
- Result: stopped at step 3000, still in stage 0 (2 entities, direct only, no filler); train loss
  ~0.6 (= guessing between the two values). Real templates put 1-3 words between entity and value
  and link sentences also contain entities, which the probe did not.
- Action: capacity/depth sweep on the 2-entity direct task.

## Diagnostic sweeps after attempt 2 (2026-09-18, scratch scripts, 2-entity direct task)

All attention-only variants stayed at the ~55% "answer any value present" plateau (chance for
2 entities = 50% + collisions) for 2500-3000 steps:

| Variant | Train acc at end |
|---|---:|
| d_model 128, 8 heads, 2 enc | 0.57 (stopped at 1250) |
| d_model 64, 4 enc layers | 0.57 (stopped at 1500) |
| d_model 64, lr 3e-4, batch 128 | 0.56 (stopped at 750) |
| d_model 128, 3 enc, lr 3e-4 | 0.55 (stopped at 1000) |
| bare "e v" pairs, batch 32 | 0.57 |
| pairs + real question template | 0.57 |
| real value sentences, no links | 0.55 |
| real value + link sentences | 0.56 |
| multi-query (all entities per lecture), 2 and 4 entities | 0.56 / 0.41 |

Information-flow check at init: output depends on both query entity and memory, so no plumbing bug.
Conclusion: optimization plateau in learning entity->value binding, independent of template
complexity, width, depth, lr and batch size.

Fix: short causal depthwise convolution (kernel 6) in encoder blocks + multi-query training.
12 entities, direct, no filler: train acc 0.29 -> 0.43 -> 1.00 at steps 250/500/750.
The entity-count curriculum was dropped: more entities per lecture gives more contrasting
queries and learns faster.

## ced_baseline attempt 3: conv encoder + multi-query, 3 decoder layers (2026-09-18)

- Files: `ced_baseline_attempt3_conv_dec3_seed1.jsonl`, `ced_baseline_attempt3_conv_dec3.log`
- Direct lookup solved: stage 0 passed at step 786; held-out direct accuracy 99.5-100% from step
  1000 on the full task (18 fillers), without having trained on filler.
- One-hop stuck: held-out one-hop 27-33% from step 1000 to ~4250 (stopped). Two-hop never trained.
- Hypothesis: one-hop needs two chained cross-attention lookups and two-hop three, and a 3-layer
  decoder leaves no slack. Next: 5 decoder layers.

## Diagnostic: 5 decoder layers, direct + one-hop from the start (2026-09-18)

- File: `diag_dec5_onehop.log` (scratch script, 12 entities, no filler, multi-query)
- Train accuracy at step 750: direct 0.28, one-hop 0.29. Stopped. With both question types
  from the start the deeper model had not even learned direct lookup, so decoder depth is not a
  quick fix.
- Next: generative chain decoder (`configs/ced_baseline_gen.yaml`). The decoder emits every
  intermediate entity and then the value, so each generation step is a single lookup.

## snced_compare attempt 1: compiler under-trained (2026-09-18)

- File: `snced_compare_attempt1_seed1.log`
- Stage A (frozen encoder, MLP compiler over mean/max-pooled chunk states, 600 steps, cosine LR)
  reached loss 3.03 at step 500 (uniform guessing ~9.5). Stopped before stage B: a compiler this
  weak would make the notes-only comparison a test of an under-trained component.
- Change: stage A now trains until held-out exact chunk extraction >= 99% (checked every 200
  steps, cap 6000 steps), mirroring Rule 2 for the notes path. If the cap is hit, the run
  continues and the report states the extraction rate.

## snced_compare attempt 2: notes re-encoded by the frozen lecture encoder (2026-09-18/19)

- Files: `results/raw/snced_compare_attempt2_shared_encoder/`, `results/tables/snced_vs_ced_attempt2_shared_encoder.md`,
  `snced_compare_attempt2_shared_encoder_seed1.log`, figures `*_attempt2.png`
- Compiler passed its gate (99.69% held-out extraction at step 200). Decoder fine-tuned 1500 steps on detailed +
  predicted-note memory; the notes-path loss plateaued at ~2.19 from step 800.
- Result (seed 1): baseline 100 / 99.8 / 99.4% at 548 / 939 / 1529 tokens; notes-only 32 / 31 / 31%;
  gold notes 32 / 32 / 30%; notes + fallback 74 / 84 / 92% (fallback rate 59 / 77 / 89%).
- Diagnosis: gold notes fail as badly as predicted notes, so the failure is reading notes, not compiling them.
  The frozen encoder, trained only on lecture prose, gives packed note sequences unusable memory states.
- Change (attempt 3): a dedicated note encoder turns each note into one memory slot, trained jointly with the
  decoder, plus a gate that gold-note accuracy must reach >= 97% on validation before evaluation.

## BABILong real-model pilot: stopped at 1.5B (2026-09-19/20)

- Files: `babilong_pilot_1k.log`, `babilong_1p5b_gate.log`, `babilong_pilot_1k_{0,1}.jsonl`,
  `results/raw/babilong_pilot/`
- Setup: BABILong 1k (bAbI facts hidden in PG19 prose), 25 questions per cell, CPU.
- Qwen2.5-0.5B-Instruct, all four memory arms: full context 48% (qa1) / 36% (qa2);
  RAG top-k 48 / 16; summary 20 / 16; SNM notes 36 / 4. Notes used 96 prompt tokens vs 710
  and answered 3.4x faster, but lost on quality, badly so on two-fact questions.
- Qwen2.5-1.5B-Instruct full-context gate: 48% (qa1) / 32% (qa2) - no better than 0.5B, and far
  below the >= 70-80% agreed gate. Wrong answers are plausible-but-incorrect rooms, i.e. genuine
  failure to track the last location, not a formatting or scoring artifact.
- Decision (pre-agreed): stop the real-model benchmark at this model size. Comparing memory
  formats under a baseline this weak violates Rule 2.
- Note: bfloat16 on this CPU is ~26x slower than float32 (no AMX), so model size here is capped by
  RAM (1.5B fp32 = 6.0 GB resident) rather than by patience.
- Next: needs a stronger base model (7B+) and a GPU before RULER / BABILong / LongMemEval can say
  anything about SN-CED.

## span_compiler attempt 1: memorised the substitution pool (2026-09-20)

- Files: `results/raw/span_compiler_attempt1_poolmemo/`, `span_compiler_attempt1_poolmemo.log`
- Setup: BIO span tagger over frozen MiniLM; training substituted every bAbI name/place/object with
  entries from fixed 40-70 word pools (NAME_POOL / PLACE_POOL / OBJECT_POOL).
- Result: retention 89/80/86% when evaluation renamed documents using THOSE SAME pools, but
  10/9/13% on the original bAbI cast (precision 97-100%, recall 12-15%).
- Diagnosis: the "unseen entities" evaluation was not unseen - it drew from the training pools. The
  genuinely unseen vocabulary was bAbI's own cast, which training had substituted away. The tagger
  learned pool membership, not syntax, so it is not open-vocabulary.
- Fix (attempt 2): sample substitution entities from a large dynamic vocabulary mined from WikiText,
  split into disjoint train/eval halves; keep bAbI's original cast in the training mix; evaluate on
  (a) the original cast, (b) held-out entity strings from neither source.

## snced_general attempt 1: the reference path was never trained (2026-09-20)

- Files: `snced_general_attempt1_untrained_reference.log`, `snced_general_attempt1_seed1.jsonl`
- The composite loss computed the detailed-memory path only under `no_grad` (as the KL target) and
  inside the routed path, where the router fired 0% of the time. So the full-context branch got no
  task gradient at all: it sat at 11.2% (chance is 12.5%) through 2250 steps while the notes path
  climbed to 23%.
- Consequence: the sufficiency term was training the notebook to imitate an untrained, near-random
  reference - and Rule 2 was broken, since nothing can be compared against a path that cannot answer.
- Fix: added `detail_task` (weight 1.0) so the reference path trains alongside the notes path; the
  KL target is now detached from the trained reference rather than from noise.

## Lost checkpoint: ced_baseline_seed1.pt overwritten by a smoke run (2026-09-20)

- While adding `--device` support I ran `end_to_end.py --max-steps 2` several times to check the
  flag. The script saves a checkpoint at the end of every run, so the healthy 100% seed-1 baseline
  was replaced by a 2-step model that answers "e20" to everything.
- Detected when warm-starting the general model produced 0.000 accuracy on the detailed path with
  every component frozen; the same checkpoint failed in the original baseline harness too.
- Seeds 2 and 3 are intact (verified 100% on 60 held-out questions). No published result is
  affected: snced_compare, router_calibration, adversarial_eval and timing_pass all load an SNCED
  checkpoint over the backbone, so the clobbered weights were never used in a reported number.
- Guard added: `end_to_end.py` refuses to overwrite an existing checkpoint from a run of under
  1000 steps. Seed 1's baseline needs retraining if a three-seed baseline set is wanted again.
