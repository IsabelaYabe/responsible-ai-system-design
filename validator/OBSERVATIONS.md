# OBSERVATIONS

Live record of what we actually observe running the **LLM 3 validator** PoC
(`notebooks/validator_judge_poc.ipynb`). Decisions live in `DECISIONS.md`; this file is for results,
surprises, and failure modes — the raw material for the doc's §10 iteration log and §9 metrics.

_Last updated: 2026-06-24_

---

## How to use this file

After each notebook run, append an entry under **Run log** with: the date, the model, what changed
since last time, and what you saw. Note anything that deviated from the **expected** column in
`DECISIONS.md` D6 — those deviations are the interesting findings.

---

## Stage-by-stage observations

### Stage 4 — Decompose & route
- **Run 2 (2026-06-22):** produced **5** atomic claims (4 `context`, 1 `world_knowledge`), not the 3
  I'd guessed. Decomposition worked well: split "single man **with** a large fortune" into two atomic
  claims (1, 2), resolved pronouns ("He" → "Mr. Bingley"), and pulled an *implicit* claim out of the
  apposition "his eldest daughter, Jane" (claim 4: "Jane is Mrs. Bennet's eldest daughter").

### Stage 5 — Per-claim verdict — first run (Run 3, 2026-06-22)
Expected vs. actual (`claude-sonnet-4-6`, N=5 @ temp 0.8):
| # | claim | grounding | expected | actual | verb | N-sample agree |
|---|-------|-----------|----------|--------|------|----------------|
| 1 | Bingley is a single man | context | Supported | **Partially supported** | 0.70 | 1.00 |
| 2 | Bingley has a large fortune | context | Supported | **Partially supported** | 0.70 | 1.00 |
| 3 | Bingley already engaged to Jane | context | Contradicted (or Unverifiable) | **Unverifiable** | 0.99 | 1.00 |
| 4 | Jane is Mrs. Bennet's eldest daughter | context | Unverifiable | **Unverifiable** ✓ | 0.97 | 1.00 |
| 5 | Published in 1813 | world_knowledge | Unverifiable | **Unverifiable** ✓ | 1.00 | — |

Findings:
1. **Grounding held (the headline success):** claim 4 (true in the real book, absent from the passage) came
   back `Unverifiable`, not `Supported` — no memory leak. The anti-memory rule worked.
2. **Seeded "Contradicted" came back `Unverifiable`:** the passage is *silent* on Bingley's engagement (never
   names Jane), so strict grounding = Unverifiable. Our seed conflated *absence* with *contradiction* — to test
   the `Contradicted` label we need a claim the passage *explicitly* opposes.
3. **Claims 1–2 → `Partially supported`:** these atomic claims still bundle a fact + a causal "excited because",
   so only the factual half is grounded → a Stage-4 atomicity gap, surfaced as a partial + verbalized 0.70.
4. **Uncertainty comparison inverted the doc's prior:** N-sample agreement is saturated at 1.00 everywhere
   (labels never flip, even at temp 0.8), while verbalized confidence is *more* discriminating (0.70 on the
   partials vs ~1.0 elsewhere). Here the "overconfident baseline" carried more information than N-sample.

### Stage 6 — Aggregation — done (Run 4)
- Given Run 3 verdicts (2× Partially supported, 3× Unverifiable — **no Supported, no Contradicted**),
  worst-case aggregation → answer-level not-fully-supported → UI **Hedged**, *not* "Not reliable"
  (which needs a `Contradicted`). The seeded error softened from Not-reliable to Hedged — see Run 3
  finding 2. To produce a clean "Not reliable", seed an unambiguous contradiction.

### Stage 7 — Uncertainty + UI mapping — done (Run 4)
- Run 3 result: N-sample agreement **saturated at 1.00** (labels stable under resampling at temp 0.8);
  verbalized confidence was the *more* discriminating signal (0.70 partials vs ~1.0). So the two signals
  **diverged** — the §7 "diverge → reportable finding" case, and in the unexpected direction.
- Open question: does N-sample need higher temp / larger N / a finer sub-label signal to flex, or is
  label-stability just genuine here? Needs harder/borderline claims to tell.
- UI uses the verbalized signal; expected UI state for the current answer: **Hedged**.

---

## Run log

### Run 0 — scaffold only — 2026-06-21
- Cells 0–3 built (deps, config, `LLMClient`, mocked inputs). Ollama backend removed.
- Validator stages 4–7 not yet implemented.
- _Result:_ pending first execution in Colab.

### Run 1 — Stage 4 drafted — 2026-06-22
- Judge model switched Haiku → `claude-sonnet-4-6` (D3).
- Stage 4 (decompose & route) drafted; stages 5–7 not yet built.
- _Result:_ pending first execution in Colab.

### Run 2 — Stage 4 first execution — 2026-06-22
- Model: `claude-sonnet-4-6`. First real run of `decompose_and_route(GENERATED_ANSWER)`.
- Output (5 claims):
  | # | claim | grounding |
  |---|-------|-----------|
  | 1 | Mrs. Bennet is excited because Mr. Bingley is a single man | context |
  | 2 | Mrs. Bennet is excited because Mr. Bingley has a large fortune | context |
  | 3 | Mrs. Bennet is especially pleased that Mr. Bingley is already engaged to Jane | context |
  | 4 | Jane is Mrs. Bennet's eldest daughter | context |
  | 5 | Pride and Prejudice was first published in 1813 | world_knowledge |
- Matched expectations? **Mostly — good deviations.** Got 5 claims, not 3: it correctly atomized the
  compound "single man with a large fortune" and decontextualized the pronoun/apposition. Routing was
  correct (4 context, 1 world_knowledge).
- Notes / next step: claim 4 is an accidental but useful grounding test for Stage 5 (true-in-world,
  absent-from-passage → should be Unverifiable, not Supported). Build Stage 5 (grounded per-claim
  verdict) next.

### Run 3 — Stage 5 first execution (verdicts + both uncertainty signals) — 2026-06-22
- Model: `claude-sonnet-4-6`. N=5 @ temp 0.8. Anti-memory prompt uses the "true or false" clause.
- Per-claim output:
  | # | claim | grounding | verdict | verbalized_conf | nsample_agreement | entropy |
  |---|-------|-----------|---------|-----------------|-------------------|---------|
  | 1 | Bingley is a single man | context | Partially supported | 0.70 | 1.00 | 0.00 |
  | 2 | Bingley has a large fortune | context | Partially supported | 0.70 | 1.00 | 0.00 |
  | 3 | Bingley already engaged to Jane | context | Unverifiable | 0.99 | 1.00 | 0.00 |
  | 4 | Jane is Mrs. Bennet's eldest daughter | context | Unverifiable | 0.97 | 1.00 | 0.00 |
  | 5 | Published in 1813 | world_knowledge | Unverifiable | 1.00 | — | — |
- Findings: (1) grounding held — no memory leak on claim 4; (2) seeded "Contradicted" → Unverifiable
  (passage silent, not opposing); (3) claims 1–2 Partially supported (Stage-4 atomicity gap: fact +
  causal "because"); (4) N-sample agreement saturated at 1.00, verbalized more discriminating —
  signals diverged in the unexpected direction.
- Next step: (a) add an *unambiguous* contradiction to test the `Contradicted` label; (b) decide whether
  to flex N-sample (temp/N/finer signal); (c) cosmetic — entropy prints "-0.00". Build Stage 6 next.

### Run 4 — full thin slice, end-to-end — 2026-06-22
- Model: `claude-sonnet-4-6`. Ran decompose → route → verdict → aggregate → UI map on the seeded example.
- Answer-level verdict: **Unverifiable** (worst-case). UI state: **⚠️ HEDGED**.
- Both uncertainty gates agree here: verbalized 0.97 and N-sample agreement 1.00 → neither flags high
  uncertainty, so flipping `UNCERTAINTY_SIGNAL` does not change the UI state on this example.
- **Thin slice complete** (D1): mocked inputs → 5 claims → grounded verdicts + 2 uncertainty signals →
  worst-case → Hedged. The seeded hallucination surfaced as **Hedged**, not Not-reliable — exercising the
  **Not reliable** path needs the unambiguous-contradiction stimulus (P2-2).
- Next: capstone `validate()` to chain the stages; then Phase 2 (P2-1, P2-2).

### Run 5 — capstone re-run: output CHANGED (non-determinism) — 2026-06-22
- Same inputs, single-entry `validate()`. Result differed from Run 3/4:
  - Decomposition returned **4** claims, not 5 — it *merged* "engaged to her eldest daughter Jane" into one
    claim (Run 3 had split "engaged to Jane" + "Jane is eldest daughter").
  - Claim 3 verdict flipped **Unverifiable (Run 3) → Contradicted (Run 5)** (verb 0.95, agree 1.00).
  - Answer-level: **Contradicted → ⛔ NOT RELIABLE** (Run 4 was Unverifiable → Hedged).
- Two findings:
  1. **The pipeline is non-deterministic even at temperature 0** — decomposition runs at temp 0 yet changed
     granularity/phrasing. LLM APIs are not bit-deterministic; **a single run is not evidence.**
  2. **N-sample agreement gave false certainty.** Within each run it was 1.00 (5/5 identical), yet the
     verdict flipped *across* runs — within-run resampling consistency ≠ reliability. This is the §5
     warning ("self-consistency measures consistency, not correctness") made concrete: claim 3 is genuinely
     *borderline* (absence vs. contradiction), per-call consistency hides that, and the instability
     propagates all the way to the user-facing UI state (Hedged ↔ Not reliable).
- Upstream cause: much of the verdict flip traces to **decomposition variance** — the merged, richer claim 3
  gave the judge more to contradict — so stabilizing Stage 4 (P2-1) also stabilizes downstream verdicts.
- Implications: (a) **measure** run-to-run variability (run `validate()` K times, tally the verdict/UI
  distribution) *before* tuning — this is the §8–9 meta-eval the doc says earns TRL 3; (b) prompt levers can
  reduce (not eliminate) variance: tighter decomposition atomicity, and an explicit absence-vs-contradiction
  tie-break in the verdict prompt.

### Run 6 — P2-3 variability baseline (current prompts, K=10) — 2026-06-22
- `validate()` ×10, same input, `n_samples=1`.
- **UI state: Hedged 4 / Not reliable 6 (40/60).** Claim count: 5 claims ×9, 4 claims ×1.
- Verdict tuples:
  - 5× (Partially, Partially, **Contradicted**, Unverifiable, Unverifiable) → Not reliable
  - 4× (Partially, Partially, **Unverifiable**, Unverifiable, Unverifiable) → Hedged
  - 1× (Partially, Partially, **Contradicted**, Unverifiable) [4 claims] → Not reliable
- **Instability is isolated to claim 3** (engaged-to-Jane), flipping Contradicted↔Unverifiable ~6/4 — it
  sits exactly on the absence-vs-contradiction boundary (D16). Claims 1–2 (Partially) and 4–5 (Unverifiable)
  are stable; decomposition is mostly stable (5 claims 90%; the Run-5 4-claim case is the 10% tail).
- Re-prioritization: **D16 (conservative tie-break) is the targeted high-value fix**; P2-1 (decomposition
  stability) is secondary here (~10% of the variance). **Baseline to beat: UI flips 40/60.**

### Run 7 — D16 conservative tie-break applied; K=10 ×2 — 2026-06-22
- After adding the absence-vs-contradiction tie-break to `SYSTEM_VERDICT`, two K=10 runs:
  - Run A: Not reliable 7 / Hedged 3 (claim count 5×7, 4×3).
  - Run B: Not reliable 6 / Hedged 4 (claim count 5×6, 4×4).
- vs Run 6 baseline (6/4): **no meaningful improvement** — claim 3 still resolves Contradicted the majority
  of the time; the conservative rule did not bind.
- Confirmed: the `SYSTEM_VERDICT` cell was re-executed before `validate()`, so D16 was genuinely exercised:
  - **Why it didn't bind:** the model treats "Mrs. Bennet thinking of marrying him off" as *directly
    entailing* "not engaged", so the rule's "directly entails the opposite" clause still licenses
    `Contradicted`. The wording didn't change the model's classification.
  - **Measurement noise:** the two post-edit runs (7/3, 6/4) differ by as much as either differs from
    baseline — K=10 carries ~±15% noise on these proportions, too small to resolve a small effect.
    Decomposition variance is also bigger than Run 6 implied (4-claim case now 3–4/10, not 1/10).
- **Reframing:** claim 3 is genuinely *borderline* (reasonable judges — and the model across runs — split).
  A ~50/50 flip may be appropriate *uncertainty*, not a verdict bug. The real gap: verbalized (0.95) and
  within-run N-sample (1.00) both reported "confident" and failed to flag it; the only signal that caught
  the instability is **cross-run dispersion** (this P2-3 loop). → candidate fix **P2-4**.

### Run 8 — temperature diagnostic — 2026-06-23
- Verdict on a *fixed* claim ("Mr. Bingley is already engaged to Jane") ×10 at temp 0.0 / 0.8 / 1.0:
  **Contradicted 10/10 at every temperature.**
- High-entropy control ("name an animal") ×8: temp 0.0 → mixed (Cat/Dog); temp 1.0 → mixed (Cat/Dog).
- Conclusions:
  1. **Temperature IS functional on `claude-sonnet-4-6`** (the animal control varies). The verdict's
     zero-variation is because its distribution is *peaked* on fixed input (cause b), not because
     temperature is ignored. The deprecation worry doesn't bite on Sonnet 4.6.
  2. **temp=0 is NOT deterministic** (animals varied at temp 0) — Anthropic output is non-deterministic
     regardless of temperature.
  3. **Within-run, temperature-based N-sample is structurally inert for peaked verdicts** — resampling a
     confident verdict yields no diversity, so agreement is always ~1.00; it cannot detect instability.
     The reliability signal that works is *cross-run* stability (Run 6/9), not within-run resampling.

### Run 9 — P2-1 (fact/causal split) + passage naming fix — 2026-06-23
- Decomposition now **7 claims**; `report(validate(...))`:
  | # | claim | verdict | verb |
  |---|-------|---------|------|
  | 1 | Mr. Bingley is a single man | **Supported** | 0.99 |
  | 2 | Mr. Bingley has a large fortune | **Supported** | 0.99 |
  | 3 | Mrs. Bennet is excited because Mr. Bingley is a single man with a large fortune | **Supported** | 0.97 |
  | 4 | Mr. Bingley is already engaged to Jane | **Contradicted** | 0.95 |
  | 5 | Jane is Mrs. Bennet's eldest daughter | Unverifiable | 0.95 |
  | 6 | Mrs. Bennet is especially pleased that Mr. Bingley is already engaged to Jane | **Contradicted** | 0.97 |
  | 7 | Pride and Prejudice was first published in 1813 | Unverifiable | 1.00 |
- Variability K=10: **UI = Not reliable 10/10; 7 claims 10/10; one identical verdict tuple ×10. The 40/60
  flip is GONE — the pipeline is now stable across runs.**
- **This confirms the instability was upstream decomposition-phrasing variance, not verdict sampling.**
  Clean, consistent claim phrasings (P2-1) + a complete passage (naming Bingley) → consistent verdicts →
  a stable UI verdict. P2-1 was the right fix (earlier downplaying corrected).
- Wins: facts 1–2 now clean `Supported` (name grounded); causal claim 3 `Supported` (passage grounds the
  excitement); the seeded engagement hallucination is **correctly and stably** flagged `Contradicted` →
  **Not reliable** (the intended catch). The earlier "borderline" flip was largely an artifact of noisy
  inputs, not genuine verdict ambiguity.
- Caveat: 10/10 is strong but not proof of 100% — a larger K could still surface rare flips.

### Run 10 — v2 (selective prediction), full pipeline — 2026-06-24
- Clean v2 notebook (`validator_judge_poc_v2.ipynb`). Gold set **7/7 correct → AUROC undefined** (no errors);
  risk-coverage flat (committed_acc 1.00 at every τ); chosen τ=0.95 (unconstrained). Confirms verbalized
  confidence **cannot be evaluated on a zero-error set** — *not* evidence the score is unusable.
- Capstone: 7 claims → **Not reliable** (seeded contradiction caught). Variability K=10: **Not reliable 10/10**,
  identical verdict tuple ×10 — fully stable.
- Decision: **expand the gold set with harder items before judging verbalized confidence** (a score needs
  judge errors to have a measurable AUROC). Multi-passage gold set added (Ch.1 + Ch.3); labels pending human
  verification.

### Run 11 — v2 selective prediction on the expanded gold set — 2026-06-24
- 24-item, 4-passage gold set (Ch.1 + 3 verbatim Ch.3 excerpts), 5 hard items. Judge accuracy **22/24 = 0.92**.
- **AUROC = 0.95** — verbalized confidence separates correct from incorrect. The **2 MISSes both sit at the
  LOWEST confidence (verb 0.85)**; every correct verdict is ≥ 0.85, most 0.95–1.00. The cheap score *is* a usable
  abstention signal — the Run-10 "undefined AUROC" was the tiny-easy-set artifact, not a property of the score.
- Risk-coverage: clean operating point at **τ=0.90 → committed accuracy 1.00 at coverage 0.83** (abstaining on the
  bottom 17% removes exactly the 2 errors). `pick_threshold(target=0.90)` chose τ=0.85 (commit all, acc 0.92);
  raising the target to ~0.95 lands τ=0.90 — a risk-coverage operating-point choice.
- Both MISSes are borderline/inference items where the gold label is itself debatable ("Bingley plans to marry a
  daughter" → judge Contradicted vs gold Unverifiable; "Darcy left early" → judge Unverifiable vs gold
  Contradicted), and the judge was appropriately LESS confident there — consistent with §9 (don't expect the
  validator to exceed human–human agreement).
- Capstone: answer → Not reliable (seeded contradiction). Variability K=10: **Not reliable 10/10**, stable.
- Verdict: **verbalized confidence validated as a usable abstention score on this set; keep it (D19), do not
  retire it.** Caveat: 24 items is still small, and the 2 errors coincide with annotation-debatable items.

---

## Closing summary — PoC characterization (2026-06-24)
The §10 TRL-3 deliverable: where the validator is reliable, where it degrades, and the validity boundary.
Notebook: `notebooks/validator_judge_poc_v2.ipynb`.

**What it reliably does (the "works" zone):**
- Decomposes an answer into atomic claims, routes by grounding source, and produces a per-claim verdict
  **grounded against the passage** (entailment, not memory) — verified by the claim "Jane is Mrs. Bennet's
  eldest daughter" (true in the book, absent from the passage) coming back `Unverifiable`, not `Supported`.
- Worst-case aggregation → 3-way UI (Valid / Not reliable / Hedged), **stable across runs** (Run 9/11: the
  seeded contradiction → **Not reliable 10/10**).
- **Verbalized confidence is a usable abstention score** (AUROC **0.95** on a 24-item gold set with hard
  items); the commit/abstain threshold is tuned on data via risk–coverage (clean point τ=0.90 → 100%
  committed accuracy at 83% coverage).

**Where it degrades / boundaries (failure modes):**
- **LLM non-determinism**: outputs vary run-to-run even at `temperature=0`; reliability must be read as a
  *distribution* (cross-run stability), never from one run (Run 5).
- **Within-run N-sample is inert** for peaked verdicts (temperature gives no diversity) — retired (Run 8, D5).
- Most apparent "judge uncertainty" was **upstream decomposition-phrasing noise**, fixed by cleaner
  decomposition + complete grounding (Run 9, P2-1) — not verdict sampling.
- **Absence-vs-contradiction borderline** claims are genuinely ambiguous; the judge (like human annotators)
  splits, and is appropriately *less* confident there (§9 IAA; Run 7/11).

**What is mocked / out of scope (validity boundary):**
- Upstream LLMs 1 / 1.2 / 2 are mocked; the grounding passage is curated (real system: bounded retrieval +
  FAISS, D15). `world_knowledge` claims short-circuit to `Unverifiable` (no web search). Judge≠generator
  family not enforced (generators mocked, D13). Gold set is tiny / single-annotator — calibration and τ are
  **illustrative, not production-grade** (D18/D19).

**Carried forward (future work, not blocking the PoC):** larger multi-annotator gold set + IAA; per-source
thresholds; real bounded retrieval; real generators with judge≠generator family; finer calibration
(reliability diagram / ECE); cross-run dispersion as an offline signal (P2-4); a stronger uncertainty score
if needed (e.g. Yadkori-style iterative-prompting MI).

**Bottom line:** the validator behaves as intended on the comprehension/contextualization path — grounded,
uncertainty-aware, and honest about what it cannot verify — with the reliability boundary explicitly
characterized rather than assumed.

<!--
Template for future entries:

### Run N — <one-line summary> — <date>
- Model: claude-sonnet-4-6
- Changed since last run: ...
- Per-claim output:
  | # | claim | grounding | verdict | verbalized_conf | nsample_agreement | entropy |
  |---|-------|-----------|---------|-----------------|-------------------|---------|
- Answer-level verdict / UI state:
- Matched expectations? deviations:
- Notes / failure modes / next step:
-->
