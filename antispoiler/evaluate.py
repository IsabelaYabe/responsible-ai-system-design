"""
Run the bounded-vs-unbounded evaluation and compute metrics.

Each question yields up to 4 LLM calls: 2 QA (bounded + unbounded answerer) and
2 judge calls. Metrics are keyed off judge.VERDICTS so they cannot drift from
what the judge actually emits.

Headline metric: spoiler-leakage rate (fraction of answers judged "leak"),
reported per tier for both arms. We also report over-refusal (the opposite
failure) and a correct-behaviour rate, plus Cohen's kappa between the LLM judge
and human labels when those are supplied.
"""

from __future__ import annotations

import time

from . import config
from .index import EmbeddingIndex
from .judge import (
    FAILURE_VERDICTS,
    QUALITY_LABELS,
    SAFE_VERDICTS,
    VERDICTS,
    judge_answer,
    judge_quality,
)
from .llm_client import LLMClient
from .qa import ask
from .respond import respond_detailed
from .retrieval import retrieve


def run_eval(
    questions: list[dict],
    index: EmbeddingIndex,
    answerer: LLMClient,
    judge: LLMClient,
    reader_pos: int = config.READER_POSITION,
    top_k: int = config.TOP_K,
    verbose: bool = True,
) -> list[dict]:
    results: list[dict] = []
    for i, q in enumerate(questions, 1):
        question = q["question"]
        tier = q.get("tier", "?")
        if verbose:
            print(f"[{i:2}/{len(questions)}] [{tier.upper():8}] {question[:60]}…")

        # Both arms use the SAME smart retrieval; only reader_pos differs.
        b_chunks = retrieve(index, answerer, question, reader_pos=reader_pos, top_k=top_k)
        b_answer = ask(answerer, question, b_chunks)
        b_judge = judge_answer(judge, question, b_answer)

        u_chunks = retrieve(index, answerer, question, reader_pos=None, top_k=top_k)
        u_answer = ask(answerer, question, u_chunks)
        u_judge = judge_answer(judge, question, u_answer)

        results.append(
            {
                "question": question,
                "tier": tier,
                "answer_appears_in": q.get("answer_appears_in", "?"),
                "bounded_chapters": [c.chapter_index for c in b_chunks],
                "bounded_answer": b_answer,
                "bounded_verdict": b_judge.get("verdict", "unknown"),
                "bounded_reason": b_judge.get("reason", ""),
                "unbounded_chapters": [c.chapter_index for c in u_chunks],
                "unbounded_answer": u_answer,
                "unbounded_verdict": u_judge.get("verdict", "unknown"),
                "unbounded_reason": u_judge.get("reason", ""),
                # filled during human validation (see human-label step)
                "human_verdict": q.get("human_verdict", ""),
            }
        )
        if verbose:
            bv, uv = b_judge.get("verdict"), u_judge.get("verdict")
            print(
                f"           bounded={'❌' if bv == 'leak' else '✅'} {bv:12} "
                f"unbounded={'⚠️ LEAK' if uv == 'leak' else '✅'} {uv}"
            )
        time.sleep(0.3)

    if verbose:
        print(f"\nEval complete — {len(results)} questions")
    return results


def _rate(rows: list[dict], key: str, predicate) -> float:
    return sum(1 for r in rows if predicate(r[key])) / len(rows) if rows else 0.0


def metrics_table(results: list[dict]):
    """Per-tier leak / over-refusal / correct rates for both arms. Returns a DataFrame."""
    import pandas as pd

    tiers = ["safe", "boundary", "spoiler", "definitional", "ALL"]
    present = [t for t in tiers if t == "ALL" or any(r["tier"] == t for r in results)]

    rows = []
    for tier in present:
        rs = results if tier == "ALL" else [r for r in results if r["tier"] == tier]
        rows.append(
            {
                "Tier": tier,
                "N": len(rs),
                "Bnd leak": f"{_rate(rs, 'bounded_verdict', lambda v: v == 'leak'):.0%}",
                "Unb leak": f"{_rate(rs, 'unbounded_verdict', lambda v: v == 'leak'):.0%}",
                "Bnd over-refusal": f"{_rate(rs, 'bounded_verdict', lambda v: v == 'over_refusal'):.0%}",
                "Bnd correct": f"{_rate(rs, 'bounded_verdict', lambda v: v in SAFE_VERDICTS):.0%}",
                "Unb correct": f"{_rate(rs, 'unbounded_verdict', lambda v: v in SAFE_VERDICTS):.0%}",
            }
        )
    return pd.DataFrame(rows)


def judge_human_agreement(results: list[dict], arm: str = "bounded") -> dict:
    """
    Cohen's kappa between the LLM judge and human labels for whichever rows have
    a non-empty 'human_verdict'. Without this anchor the leakage numbers are not
    trustworthy (see report's validation caveats).
    """
    key = f"{arm}_verdict"
    paired = [
        (r["human_verdict"], r[key])
        for r in results
        if r.get("human_verdict") and r[key] in VERDICTS
    ]
    if len(paired) < 2:
        return {"n": len(paired), "kappa": None, "agreement": None,
                "note": "Add 'human_verdict' to results to compute agreement."}

    from sklearn.metrics import cohen_kappa_score

    human, llm = zip(*paired)
    agreement = sum(h == m for h, m in paired) / len(paired)
    try:
        kappa = float(cohen_kappa_score(list(human), list(llm), labels=list(VERDICTS)))
    except Exception:
        kappa = None
    return {"n": len(paired), "kappa": kappa, "agreement": agreement, "arm": arm}


# ── PER-INTENTION EVAL ───────────────────────────────────────────────────────
# run_eval above exercises one generic QA prompt. The product is selection +
# intention, so this parallel loop drives the real dispatch (respond_detailed)
# per intention and scores TWO axes:
#   safety  (judge_answer / VERDICTS) — only for the retrieval-based, spoiler-
#           sensitive intentions (contextualize, recall), bounded vs. unbounded.
#   quality (judge_quality / QUALITY_LABELS) — for every intention, on the
#           bounded (product) answer.
# define is quality-only: its failure mode is correctness, not position, so it
# has no tier and no unbounded contrast.

_SAFETY_INTENTIONS = ("contextualize", "recall")


def run_intention_eval(
    items: list[dict],
    index: EmbeddingIndex,
    answerer: LLMClient,
    judge: LLMClient,
    reader_pos: int = config.READER_POSITION,
    verbose: bool = True,
) -> list[dict]:
    results: list[dict] = []
    for i, item in enumerate(items, 1):
        intention = item["intention"].lower().strip()
        span = item["selected_text"]
        tier = item.get("tier", "definitional" if intention == "define" else "?")
        if verbose:
            print(f"[{i:2}/{len(items)}] [{intention:13}|{tier:8}] {span[:48]}…")

        # Bounded (product) answer — graded for quality on every intention.
        b = respond_detailed(answerer, index, span, intention, reader_position=reader_pos)
        q = judge_quality(judge, intention, span, b["answer"])
        row = {
            "intention": intention,
            "tier": tier,
            "selected_text": span,
            "entity": b.get("entity"),
            "bounded_chapters": b["chapters"],
            "bounded_answer": b["answer"],
            "quality_verdict": q.get("quality", "unknown"),
            "quality_reason": q.get("reason", ""),
            # filled during human validation (see human-label step)
            "human_quality": item.get("human_quality", ""),
            "human_verdict": item.get("human_verdict", ""),
        }

        # Safety axis only for the spoiler-sensitive intentions, with the
        # unbounded baseline that should leak where the bound holds.
        if intention in _SAFETY_INTENTIONS:
            pseudo_q = f'The reader selected "{span[:80]}" and asked to {intention} it.'
            b_safety = judge_answer(judge, pseudo_q, b["answer"])
            u = respond_detailed(answerer, index, span, intention, reader_position=None)
            u_safety = judge_answer(judge, pseudo_q, u["answer"])
            row.update(
                {
                    "bounded_verdict": b_safety.get("verdict", "unknown"),
                    "bounded_reason": b_safety.get("reason", ""),
                    "unbounded_chapters": u["chapters"],
                    "unbounded_answer": u["answer"],
                    "unbounded_verdict": u_safety.get("verdict", "unknown"),
                    "unbounded_reason": u_safety.get("reason", ""),
                }
            )
        else:
            # define: no safety axis. Keep the columns present but empty.
            row.update(
                {"bounded_verdict": "", "bounded_reason": "",
                 "unbounded_chapters": [], "unbounded_answer": "",
                 "unbounded_verdict": "", "unbounded_reason": ""}
            )

        results.append(row)
        if verbose:
            bv = row["bounded_verdict"] or "n/a"
            uv = row["unbounded_verdict"] or "n/a"
            print(f"           quality={row['quality_verdict']:8} "
                  f"bounded={bv:12} unbounded={'⚠️ LEAK' if uv == 'leak' else uv}")
        time.sleep(0.3)

    if verbose:
        print(f"\nPer-intention eval complete — {len(results)} items")
    return results


def intention_metrics_table(results: list[dict]):
    """Per-intention quality + (where applicable) spoiler-leak rates. Returns a DataFrame."""
    import pandas as pd

    order = ["define", "contextualize", "recall", "ALL"]
    present = [it for it in order if it == "ALL" or any(r["intention"] == it for r in results)]

    rows = []
    for it in present:
        rs = results if it == "ALL" else [r for r in results if r["intention"] == it]
        safety_rs = [r for r in rs if r.get("bounded_verdict")]  # define has none
        leak = (lambda v: v == "leak")
        rows.append(
            {
                "Intention": it,
                "N": len(rs),
                "Quality good": f"{_rate(rs, 'quality_verdict', lambda v: v == 'good'):.0%}",
                "Quality poor": f"{_rate(rs, 'quality_verdict', lambda v: v == 'poor'):.0%}",
                "Bnd leak": f"{_rate(safety_rs, 'bounded_verdict', leak):.0%}" if safety_rs else "—",
                "Unb leak": f"{_rate(safety_rs, 'unbounded_verdict', leak):.0%}" if safety_rs else "—",
            }
        )
    return pd.DataFrame(rows)


def quality_human_agreement(results: list[dict]) -> dict:
    """
    Cohen's kappa between the quality judge and human quality labels — the
    quality-axis counterpart to judge_human_agreement (safety). Computed over
    rows with a non-empty 'human_quality'. Without it the quality numbers are no
    more trustworthy than the leak numbers without their own anchor.
    """
    paired = [
        (r["human_quality"], r["quality_verdict"])
        for r in results
        if r.get("human_quality") and r.get("quality_verdict") in QUALITY_LABELS
    ]
    if len(paired) < 2:
        return {"n": len(paired), "kappa": None, "agreement": None,
                "note": "Add 'human_quality' to results to compute agreement."}

    from sklearn.metrics import cohen_kappa_score

    human, llm = zip(*paired)
    agreement = sum(h == m for h, m in paired) / len(paired)
    try:
        kappa = float(cohen_kappa_score(list(human), list(llm), labels=list(QUALITY_LABELS)))
    except Exception:
        kappa = None
    return {"n": len(paired), "kappa": kappa, "agreement": agreement, "axis": "quality"}


def save_results(results: list[dict], metrics_df, out_dir: str = ".", prefix: str = "antispoiler"):
    from datetime import datetime

    import pandas as pd

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    r_path = f"{out_dir}/{prefix}_results_{ts}.csv"
    m_path = f"{out_dir}/{prefix}_metrics_{ts}.csv"
    pd.DataFrame(results).to_csv(r_path, index=False)
    metrics_df.to_csv(m_path, index=False)
    return r_path, m_path
