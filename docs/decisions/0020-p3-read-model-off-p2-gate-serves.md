# 0020: The P3 read model is off by default; the serving gate is the P2 floors gate plus the conflict model

- **Status:** Accepted (the owner, 2026-10-25, MEM2-1 comments 22:24 and 22:59, after B-E1); **MEM2-171 parked as an experiment**
- **Date:** 2026-10-25 · **Task:** MEM2-183 · **By:** Core (Harness) on the owner's decision

## Context
The P3 read model (MEM2-155, `meta.gate.read_model`) replaces the Platt probability on the rerank logit with
`p = σ(bias + Σ wᵢ·xᵢ)` over `rerank`, `cosine`, `gap` and, from MEM2-165/171, the subject-masked signals
(`cosine_masked`, `rerank_masked`), the relation-phrase match and the masked margins. It was fitted to meet the
SYN B-R2 (ii) missing-slot abstention and the B-R5 rows. B-E1 is the first end-to-end answer check on real
data: LME tune, 125 questions, Sonnet 5.5 as answerer and judge, LongMemEval rules; **descriptive** (dev/tune data,
no test look, no target changed).

## B-E1 host evidence (125 questions, accuracy)
| Arm | Accuracy |
|---|---|
| P3 gate (the read model) | **31%** |
| **P2 gate** (floors, Platt on rerank) | **78%** |
| Gate bypassed (top 10 rounds, no abstention) | 80% |
| Oracle (gold evidence) | 96% |

Per question type under the P2 gate: single-session 100%, knowledge-update 100%, multi-session 72%, temporal 53%,
abstention 10/10. Under the P3 read model the recall returned **0 rows on 30 of 38 temporal questions**: the gate
abstained on questions that have an answer. The P2 gate is within 2 points of having no gate at all, so the gate
costs almost nothing on real questions, while the P3 model, tuned on SYN, throws away most of them.

## Decision
1. **`read.p3_model: false` by default** (a config switch, inside `fp_all`). The serving gate is the **P2 floors gate**
   (Platt on the rerank logit, as `gate-lme-only.r6`) **plus the conflict model** (`bench gate-add-theta`, the base sweep
   row first). With the switch off, `recall` computes `p = σ(a·z + b)` and ignores any `read_model` in the gate file; no
   cosine is computed.
2. The P3 read model **stays in the code**: `mem calibrate --objective p3`, its fit, the masked / relation-phrase / margin
   features and their tests are kept. Turning the switch on uses it.
3. **Gate file rule:** a gate without a `read_model` section is accepted when `read.p3_model` is off, and refused when it is
   on (`mem serve` preflight, `recall`, `bench gate P3`). A gate that carries a `read_model` is accepted either way.
4. **New fingerprint.** `read.p3_model` joins the `read` section, so `fp_all` changes
   (`eaabca64…` → `55f53d2e…`, `tests/test_config.py::GOLDEN_FP_ALL`). Every gate fitted before this change is void and needs
   **re-gating** (deliverables/runbooks/p3-gate.md, Round 8). The golden test is updated.
5. **B-R2 (ii) missing-slot abstention is a known miss**, accepted by the owner on 2026-10-25: SYN 0.10 under the P2 gate
   (threshold 0.90 unchanged); real-world abstention under the P2 gate is 10/10 on LME B-E1. `bench gate P3` prints it as
   `KNOWN MISS` (not pass, not fail), like the B-R5 rows. Revisit on PG-v1.

## MEM2-171 parked: adoption rule
The relation-phrase + masked-margin features (MEM2-171, `mem.relphrase`) stay as an **experiment, off by default**. They are
adopted (the switch turned on with them in the fit) **only if, on the owner's host pass, all three hold**:
1. B-R2 (ii) improves (above the SYN 0.10 of the P2 gate);
2. **and** B-E1 tune accuracy stays **≥ 0.75**;
3. **and** LME `coverage_any` does not fall below the P2-gate value.

The **LME coverage guard** (MEM2-176) stays in the p3 fit. The answerability cross-encoder is **not built**; only if
the owner asks again.

## Consequences
- Serving does less work (no per-candidate cosine, no masked query embedding) and answers like the 78% arm.
- SYN B-R2 (ii) stays at 0.10 until a read model passes the adoption rule; SYN is a synthetic proxy and B-E1 says it
  mislead here.
- Not a test look: no number above is a held-out result.
