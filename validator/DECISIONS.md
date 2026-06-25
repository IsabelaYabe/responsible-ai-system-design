# DECISIONS

Live record of design decisions for the **LLM 3 validator** proof-of-concept. Current notebook:
`notebooks/validator_judge_poc_v2.ipynb` (clean rebuild; `validator_judge_poc.ipynb` is the earlier
scratch version). See §10 of the design doc for why we keep this.

Each decision lists the rationale, the relevant design-doc section, and a status:
**Implemented** (in the notebook now) or **Planned** (agreed, not yet built).

_Last updated: 2026-06-24_

---

## Scope & setup

### D1 — Prototype scope: full thin vertical slice — Implemented
Build the whole pipeline end-to-end on **one** worked example
(decompose → route → per-claim verdict → worst-case aggregate → uncertainty → 3-way UI map),
rather than only the minimal grounded paraphrase judge.
**Why:** the goal is to understand the *mechanism* as a whole, not just one check. A thin slice
exercises every stage once and de-risks the full loop (doc §11: "Proving the loop end-to-end on
one functionality first de-risks the whole thing").

### D2 — Judge is a real LLM; upstream LLMs are mocked — Implemented
LLM 3 (the validator) makes real Claude calls. LLM 1 / 1.2 / 2 (contextualization, comprehension,
spoiler gate) are **not** run — we hard-code their output (question + passage + answer to validate).
**Why:** the validator is the thing under test ("does it actually work as intended?"); the
generators are upstream and out of scope for this PoC.

### D3 — Backend + model: Anthropic / `claude-sonnet-4-6` — Implemented
Judge runs on **Sonnet** (changed from Haiku). The verdict stage is essentially NLI/entailment
("is this claim supported by the passage?") — the capability-sensitive step of the whole pipeline —
so a stronger model yields better verdicts and better-calibrated verbalized confidence. Haiku stays a fine cheaper option if cost becomes a concern (e.g. for the mechanical decompose stage).
See also **D12** (judge vs. uncertainty model) and **D13** (judge vs. generator family).
**Portability note (Run 8):** `temperature` (used for N-sample) is accepted and functional on Sonnet 4.6,
but is **removed (400)** on Opus 4.7/4.8 and Fable 5 — switching the judge to those models would break the
temperature-based sampling (use adaptive thinking + `effort` instead).

### D4 — Drop the Ollama backend — Implemented
Kept Anthropic + OpenAI in `LLMClient`; removed Ollama and its config vars.
**Why:** unused for this PoC; fewer moving parts.

### D5 — Uncertainty: verbalized confidence AND N-sample consistency, compared from the start — Resolved (Run 8): within-run N-sample retired
The doc's preferred *single-pass logprob* signal is **unavailable** (the Anthropic API exposes no
output-token logprobs), so we implement and **compare both** feasible signals on the worked example
rather than deferring N-sample:
- **Verbalized confidence (cheap, single-pass *candidate*):** one verdict call at `temperature=0`; the
  judge reports a confidence (0–1) next to its verdict. The doc treats this as a *"baseline to beat"*
  (known-overconfident), **not** a settled production signal.
- **N-sample consistency (the principled signal):** the *same* judge prompt re-run N times at
  `temperature>0`; uncertainty = label dispersion across the N samples — the **agreement fraction** of
  the modal verdict, plus the **Shannon entropy** of the label distribution.

**Why both, now:** comparing them on the example *is* the §7 experiment — and on our stack it decides an
*architecture* question, not just a metric. Running both from the start surfaces the relationship on
every run.

**Where N-sample lives is PENDING the comparison (not settled).** The doc's three tiers are: logprob
(preferred, in-backend — *unavailable* to us), verbalized (baseline-to-beat, overconfident), and
N-sample (offline validator *by default*, but **"becomes the primary signal if the API lacks logprobs,"**
§7/§12). Because we have no logprobs, the placement is conditional on the result:
- verbalized **tracks** N-sample → ship verbalized in the backend, keep N-sample **offline** (validation);
- verbalized **diverges** (stays overconfident) → N-sample becomes the **primary production signal**, in
  the backend (mitigations then: parallelize the N calls, smaller/adaptive N, cascade only on borderline
  cases, or restrict to high-stakes claim types).

So D5 does **not** yet fix the production verdict's uncertainty source — that is an output of the experiment,
not an assumption.
**Mechanics:** `LLMClient.complete()` gains a `temperature` argument (Sonnet 4.6 accepts it; we set no
`thinking`). N defaults to ~5, temperature ~0.8 — both configurable. N-sample re-samples the **same**
judge model, consistent with D12.
**Finding (Run 8): within-run N-sample is structurally inert for peaked verdicts.** Temperature is
functional on Sonnet 4.6, but a confident verdict's token distribution is peaked enough that resampling
(even at temp 1.0) yields no diversity → agreement is always ~1.00. So within-run N-sample cannot detect
instability on this setup; the reliability signal that actually works is **cross-run** stability (re-running
the full pipeline, P2-3/P2-4), not within-run resampling. This weakens N-sample as implemented and makes
verbalized confidence + cross-run stability the practical signals.
**Resolution (decided):** **retire within-run N-sample** (inert). Per-request signal = **verbalized
confidence** (one cheap pass per claim, run in parallel across claims). **Cross-run stability** = the
**offline** reliability metric (§8 harness, parallelized) — never in the request path, so its cost never
reaches the reader. Calls are parallelized with a `parallel_map` (ThreadPoolExecutor; LLM calls are
I/O-bound, so concurrency gives near-linear speedup within rate limits).

### D12 — Judge and its uncertainty estimate use the SAME model — Implemented
The verdict and its uncertainty are produced by one model. Uncertainty here is a property of the
*judge's own verdict* — the verbalized confidence it reports, and how consistent that same judge is
across N re-samples (N-sample consistency, D5). Both signals come from re-sampling this one judge.
**Why:** using a *different* model for "uncertainty" would measure **inter-model agreement**, not the
judge's confidence in its own verdict — a different quantity that breaks the calibration story (doc §9:
does *this judge's* confidence track *this judge's* accuracy?). So splitting judge ≠ uncertainty across
models is disallowed; judge + uncertainty are the same model (D3).

### D13 — Judge model family must differ from the generator (deferred) — Planned
When the upstream generators (LLM 1 / 1.2) become real, the validator should run on a *different model
family* than the generator that produced the answer being judged.
**Why:** cheap mitigation of self-preference / self-recognition bias — a judge tends to rate its own
family's output more favorably (doc §12). **Deferred:** the generators are mocked in this PoC (D2), so
there is no generator family to differ from yet. Revisit when LLM 1/1.2 are wired in. (Note this is a
*different* axis of mixing from D12: D12 forbids splitting judge vs. uncertainty; D13 requires judge ≠
generator.)

### D14 — Interaction model: select-text + feature button (no free-text prompt) — Implemented
The reader does not type a free-text question. They (1) **select a passage** and (2) click a
**feature button**; the model infers the request from `(selected text, feature)`. Selection is
capped at the reader's current position, so nothing past their current chapter can be chosen. The
four features map onto the doc's request paths / grounding sources:

| Button | Request path | Grounding source (§3) |
|--------|--------------|------------------------|
| Define | comprehension (LLM 1.2) | lexical-semantic (definition) |
| Paraphrase | comprehension (LLM 1.2) | linguistic-equivalence (vs. selected text) |
| Contextualize | contextualization (LLM 1 → spoiler gate) | context-grounded |
| Recall | in-book memory aid: recall earlier plot the reader has already read | context-grounded (vs. in-bounds passages, ≤ reader position) |

**Why:** a fixed, intent-typed action set makes the request unambiguous — the validator knows *what
kind* of help was asked for, which is exactly what tells it which grounding check to apply. It also
bounds the spoiler surface (selection can't exceed the reader's position). Our worked example uses
**Contextualize** (D6).

---

## Worked example

### D6 — Seeded contextualization example (Pride and Prejudice) — Implemented
The reader selects the line *"A single man of large fortune; four or five thousand a year…"* and
clicks **Contextualize** (D14) — a contextualization request (characters/background), so it takes
the **LLM 1 → LLM 2 spoiler gate → LLM 3** path; the comprehension features (Define/Paraphrase,
LLM 1.2) are not exercised here. (Careful: the request *path* "contextualization" is a different
axis from the claim *grounding source* "context" in D7 — the shared word is just naming overlap.)
Inputs: the selected text + the chosen feature + the in-bounds grounding passage + a generator
answer **seeded** with a mix of claim types:
| # | Claim | Grounding | Expected verdict |
|---|-------|-----------|------------------|
| 1 | Bingley is a single man of large fortune | context | Supported |
| 2 | He is already engaged to Jane | context | **Contradicted** (seeded error) |
| 3 | The novel was published in 1813 | world-knowledge | Unverifiable (no web search in PoC) |

**Why:** a known-answer case with a deliberate hallucination lets us check the validator catches a
real error (doc §8: "seed deliberate errors so detection can be measured"). Worst-case aggregation
should flag the whole answer **Not reliable**.

### D15 — Grounding context: curated passage now, bounded retrieval later — Implemented (PoC stub) / Planned (integrated)
In the PoC, `SOURCE_PASSAGE` is a curated passage that already contains the grounding for the seeded
claims. In the integrated validator it is replaced by **bounded retrieval**: the anti-spoiler
notebook's `retrieve_bounded(query, reader_pos)` + FAISS, returning the **top-k relevant chunks** from
the reader's read-so-far scope (chapters ≤ reader position, up to the current page).
**Why retrieve, not dump-everything:** the grounding handed to the judge is the *relevant* chunks, not
all read text — passing ~15 chapters per claim is costly and *hurts* entailment precision (more
distractor text to be misled by). Doc §4: "for long PDFs, retrieve the relevant chunks first." The
read-so-far scope is the spoiler *bound*; retrieval picks the relevant slice *within* it.
**Why curated in the PoC:** isolates the failure mode — a wrong verdict is then unambiguously the
judge's fault, not a retrieval miss (doc §11 staging; §8's "small curated set").
**Limitation:** the curated passage tests only the **happy path** (grounding present). Add later: feed
missing / noisy / irrelevant grounding and verify the judge falls back to `Unverifiable` instead of
guessing a verdict.
**Feature-dependent grounding (later):** Paraphrase → the selected text itself (no retrieval); Define →
dictionary + local passage; Contextualize / Recall → retrieved read-passages; world-knowledge →
external (web, §5).

---

## Stage 4 — Decomposition & routing

### D7 — Grounding-source labels: only `context` and `world_knowledge` — Implemented
Defer `paraphrase` and `lexical-semantic` for now; structure the router so they're a one-line add.
**Why:** (a) they're the only two our worked example exercises; (b) a 2-way classification is
more reliable than a 4-way one; (c) they map onto the only behavioral fork that matters in
this PoC — "groundable in the passage" (→ real verdict) vs. "not" (→ Unverifiable) — since there is
no web search. `paraphrase`/`lexical-semantic` are just additional *grounded-against-passage*
buckets; deferring them changes no verdict/aggregation machinery (doc §3).

### D8 — Decomposition is separate from verdict; routing rides with decomposition — Implemented
`decompose_and_route()` returns claims tagged by grounding source only (no verdicts). A separate
Stage 5 produces verdicts, with the passage in front of it.
**Why:** this is the doc's core reliability lever (§4) — passing the source passage to the verdict
step forces *entailment against provided text* instead of holistic recall from memory, which is the
failure mode §1 is built to avoid. Separation also keeps per-claim uncertainty clean and makes the
"decomposed vs. holistic judging" ablation (§10) trivial. Routing is a property of the claim (read
off the answer), not a truth judgment, so it belongs with decomposition.

### D17 — Prompt few-shot examples must be out-of-domain — Implemented
Any worked example embedded in a stage prompt (e.g. the fact/causal split example in `SYSTEM_DECOMPOSE`)
must use content **unrelated** to the test passage — invented characters/scenarios (a lighthouse keeper),
never Mrs. Bennet / Bingley.
**Why:** an in-domain example demonstrates the desired behavior on the very claims we are measuring, which
biases the result — the prompt effectively leaks the expected decomposition for our test case. Out-of-domain
examples teach the *pattern* without contaminating the eval (§8–9 require the harness to measure unbiased
behavior). Applies to every stage prompt, not just decomposition.

---

## Stages 5–7 — verdict, aggregation, UI mapping

### D9 — Per-claim verdict labels — Implemented
Supported / Partially supported / Contradicted / Unverifiable (doc §4). `Unverifiable` is a
first-class label, not a failure — it handles claims the judge cannot ground.

### D10 — Answer-level aggregation: worst-case — Implemented
Any non-Supported claim flags the whole answer (doc §4: "worst-case… the safe default for a
comprehension aid").

### D11 — UI mapping: 3-way (Valid / Not reliable / Hedged) — Implemented
Per doc §6: Supported + low uncertainty → Valid; Contradicted/unsupported + low uncertainty →
Not reliable; high uncertainty OR Unverifiable → Hedged.

### D16 — Verdict tie-break: conservative (silence → Unverifiable) — Tested (Run 7): did not bind
On the absence-vs-contradiction boundary, the judge must be **conservative**: use `Contradicted` only when
the passage *explicitly states or directly entails* the opposite of the claim; if the passage is merely
silent, use `Unverifiable` — even if other statements loosely suggest otherwise.
**Why:** Run 5 showed claim 3 ("engaged to … Jane") flip Unverifiable↔Contradicted across runs — it sits on
this boundary. For a comprehension aid we prefer **fewer confident "Not reliable" calls on claims the passage
doesn't actually deny**: a loose inference → Contradicted → Not reliable hurts trust more than an honest
Hedged. Sharpening the rule also cuts run-to-run flip-flopping. Applied as a Phase 2 prompt lever and
validated against the P2-3 baseline; pairs with P2-1 (decomposition stability).

### D18 — Reliability calibration via a tiny gold set (mocked-scale §8/§9) — Superseded by D19
We do **not** hardcode reliability — we **compute** it from a small **gold set** (claims with
human-assigned correct verdicts), then use it to calibrate the judge's verbalized confidence:
`calibrated_conf = verbalized_conf × reliability(grounding_source)`, where `reliability(src)` = the judge's
accuracy vs. gold on that source. The UI gate uses `calibrated_conf`.
**Why:** this shows the §8/§9 rationale end-to-end — *how* reliability is measured from data and fed back
to calibrate the cheap production signal — instead of asserting a number. An overconfident judge gets its
confidence discounted toward its measured accuracy, which can correctly demote a shaky Valid/Not-reliable
to **Hedged**.
**Mock boundary (state it in the writeup):** the gold set here is **tiny — illustrative only** (a handful of
items, a single annotator). A real application needs a much larger curated gold set with multiple annotators
and inter-annotator agreement (§9), and likely finer calibration (per verdict type, or a full reliability
diagram / ECE) rather than one per-source accuracy. World-knowledge is short-circuited (the judge doesn't
adjudicate it), so calibration doesn't apply there — those claims are `Unverifiable → Hedged` regardless.

**Rationale for the multiplicative form `cal = verb × reliability` (and what it isn't):** it is a
*shrinkage/discount* — reliability=1 → identity (trust the self-report), reliability=0 → 0 (ignore it),
monotonic between, and it can only *lower* confidence, which matches the judge's overconfidence failure mode.
It is **not** a valid probability product (verb and reliability are two estimates of the *same* event, not
independent), and one per-source scalar is coarser than proper calibration (a learned confidence→accuracy
map: Platt/temperature scaling, isotonic regression, a reliability diagram — all of which need a large gold set).

**External basis — this is the Dempster–Shafer discounting operation.** D–S evidence theory discounts an
unreliable source's belief by a reliability coefficient β∈[0,1]: `^βm = β·m + (1−β)·m_Ω` (the freed mass
goes to *ignorance* `m_Ω`). Model the verdict as a simple support function (mass `verb` on "verdict correct",
`1−verb` on ignorance) and discount with β=`reliability`: the belief in the verdict becomes exactly
**`reliability × verb` = our `calibrated_conf`**, and the discounted share flows to ignorance — i.e. "couldn't
verify" → **Hedged**. So the formula is the discounted belief mass of a simple support function, not an ad-hoc
product. The *per-context* reliability form ("contextual discounting") matches our per-grounding-source design.
- Shafer, *A Mathematical Theory of Evidence*, Princeton Univ. Press (1976) — origin of discounting + reliability coefficient.
- Mercier, Quost & Denœux, "Refined modeling of sensor reliability … using contextual discounting," *Information Fusion* (2008) — per-context reliability (≈ per grounding source).
- LLM-side support: confidence-calibration surveys list **product** (with min / harmonic-mean / learned weighting) as a recognized rule for combining confidence signals; verbalized confidence is typically overconfident and needs calibration vs. accuracy (e.g. arXiv:2410.06707).

### D19 — Reliability via selective prediction: threshold tuned on the gold set — Implemented (supersedes D18)
Instead of calibrating confidence with a multiplicative reliability factor (D18), keep the **raw verbalized
confidence as the score** and **tune the commit/abstain threshold τ on the gold set** (selective prediction /
score-based abstention): commit to a verdict when `confidence ≥ τ` (→ Valid / Not reliable), else abstain
(→ Hedged). τ is read off the **risk–coverage curve** at a target committed-accuracy, after an **AUROC** check
that the score is discriminative at all.
**Why (over D18):** this is what the design doc prescribes — §5 *"earn the threshold"* (adopt a
`confidence ≥ τ → treat as correct` rule only if the data justify it) and §9 *selective prediction /
risk–coverage*. It's grounded in selective-classification (El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017)
and score-based abstention for LLM hallucination (Yadkori et al., 2024). It optimizes the actual decision
boundary on data rather than recalibrating an intermediate score, and forces the honest AUROC question
(*is verbalized confidence even usable as an abstention score?*) instead of papering over it.
**Caveats:** (a) tiny gold set → τ is noisy (mock boundary, as in D18); (b) if the judge makes no errors on
the gold set, AUROC is undefined and τ is unconstrained — itself a finding (need a larger/harder set, or a
better score: Yadkori-style iterative-prompting MI, or cross-run dispersion P2-4).
**D18's multiplicative form (Dempster–Shafer discounting) is retained in the record as
considered-and-rejected for this PoC**, in favour of the doc-prescribed selective-prediction approach.
Implemented in `notebooks/validator_judge_poc_v2.ipynb` (§7).
**Empirical result (Run 11):** on a 24-item gold set with deliberately hard items, **AUROC = 0.95** — verbalized
confidence does separate correct from incorrect verdicts (the 2 judge errors were the lowest-confidence items,
verb 0.85). A clean risk–coverage operating point exists at **τ=0.90 (committed accuracy 1.00 at 83% coverage)**.
So verbalized confidence is a usable abstention score here and is **kept**; the earlier undefined AUROC (Run 10)
was the tiny-easy-set artifact, not a property of the score. (Selecting τ=0.85 vs 0.90 is the risk–coverage
operating-point choice, set by `TARGET_ACCURACY`.)

---

## Phase 2 experiments (after the v1 PoC)
Deferred experiments surfaced while building v1; revisit once the full thin slice runs end-to-end.

### P2-1 — Split fact from causal attribution in decomposition — Implemented (Run 9): stabilized the pipeline
Run 3 showed claims like "Mrs. Bennet is excited **because** Mr. Bingley is a single man" come back
`Partially supported`: Stage 4 left a causal "because …" welded onto the fact, so each claim is really
*fact + causal attribution*. Try a decomposition prompt that separates the **fact** ("Bingley is a single
man") from the **causal/attitudinal** claim ("Mrs. Bennet's excitement is due to X"), and check whether
verdicts get crisper (more clean `Supported`, fewer partials). Compare against the v1 decomposition.
Run 5 showed decomposition variance also *destabilizes downstream verdicts* (a different claim set flipped
the final UI state), so this targets **pipeline stability**, not just cleaner partials.
**Result (Run 9):** applied together with restoring the Bingley-naming line in `SOURCE_PASSAGE`. Facts 1–2
became clean `Supported`, the causal claim `Supported`, and the whole pipeline went from a 40/60 UI flip to
**Not reliable 10/10** — confirming the instability was upstream phrasing variance, not verdict sampling.

### P2-2 — Seed an unambiguous contradiction — Planned
Run 3's seeded "engaged to Jane" came back `Unverifiable` because the passage is *silent*, not opposing
(absence ≠ contradiction). To exercise the `Contradicted` label and the **Not reliable** UI path, seed a
claim the passage *explicitly* opposes — e.g. "Mr. Bingley is a poor man" (passage: "large fortune; four
or five thousand a year") or "Mr. Bennet had already heard about Netherfield" (passage: "Mr. Bennet
replied that he had not"). Keep the judge's strict absence→Unverifiable behavior; this only fixes the
test stimulus.

### P2-3 — Measure run-to-run variability (do this FIRST) — Planned
LLM calls are non-deterministic even at `temperature=0` (Run 5): the same input produced a different claim
set and a flipped final verdict (Hedged → Not reliable). **Before tuning any prompt**, run `validate()` on
the *same* input K times (≥10) and tally the distribution of per-claim verdicts and the final UI state.
This baseline (a) quantifies the instability, (b) is the only way to tell whether a later prompt change
actually helped, and (c) is the §8–9 meta-eval the doc says earns TRL 3. Re-measure after each prompt lever
(P2-1 decomposition, D16 tie-break). For the variability run, use a low `n_samples` — we are measuring the
*across-run* spread of the single-pass production verdict, not within-run N-sample consistency.

### P2-4 — Cross-run dispersion as an uncertainty signal — Planned
Run 5/7 showed within-run N-sample consistency (and verbalized confidence) give **false certainty** on
claims that flip *across* runs (claim 3: verbalized 0.95, N-sample agreement 1.00, yet ~50/50 across runs).
The only signal that catches this is the P2-3 loop itself — re-running the verdict (with fresh decomposition)
and measuring how often the label changes. Promote that from a diagnostic into a real uncertainty signal:
high cross-run dispersion → high uncertainty → route the answer to **Hedged** regardless of any single run's
label. Expensive (K× calls); revisit cost via the D5 mitigations (smaller K, cascade, high-stakes-only).
This reframes the claim-3 instability as something to *detect and surface*, not necessarily prompt away.
**Reframed (Run 8/9):** cross-run stability lives **offline** (§8 meta-eval on a curated set), **not** in
the per-request path — so latency is a non-issue; parallelize the K runs. Run 9 also showed that cleaning
upstream inputs (P2-1) collapses the dispersion, so cross-run stability is primarily a *development/eval
reliability metric* ("did this change help?"), not a per-request uncertainty gate.
