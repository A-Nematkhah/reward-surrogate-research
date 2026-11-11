"""In-repo mini host loop: generate → Tier0/mb → select → label → update."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from reward_triage.llm_client import LLMClient, LLMResponse
from reward_triage.pipeline import triage_candidate
from reward_triage.probes import load_probe_npz
from reward_triage.schema import CandidateRecord
from reward_triage.store import (
    LabelStore,
    ModeName,
    SeedLabel,
    SelectionLog,
    StoreRecord,
    validate_store,
)
from reward_triage.tier0 import is_hard_reject

Decision = Literal["train", "skip", "audit"]


@dataclass
class LoopConfig:
    run_id: str
    env_id: str = "RewardNavigation2D-v0"
    env_config_id: str = "2d"
    host: str = "mini_host"
    n_per_generation: int = 16
    n_generations: int = 3
    n_seeds: int = 3
    seeds: tuple[int, ...] = (0, 1, 2)
    mode: ModeName = "off"
    m_min: int = 24
    k_frac: float = 0.5
    audit_frac: float = 0.15
    temperature: float = 0.7
    prompt_id: str = "gen0_default"
    probe_path: str = "data/probes/navigation_2d.npz"
    store_path: str = ""
    dry_run: bool = True
    max_jobs: int | None = None
    ppo_timesteps: int = 50_000
    enable_early: bool = True
    experiments_root: str = "experiments"
    env_config: dict[str, Any] | None = None
    ppo_config: dict[str, Any] | None = None
    eval_config: dict[str, Any] | None = None


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _candidate_id(run_id: str, generation: int, index: int, digest: str) -> str:
    return f"{run_id}_g{generation}_{index:03d}_{digest}"


def render_prompt(template: str, *, task: str, parent_code: str = "", stats: str = "") -> str:
    return (
        template.replace("{{task}}", task)
        .replace("{{parent_code}}", parent_code or "# none")
        .replace("{{stats}}", stats or "{}")
    )


def select_decisions(
    n: int,
    *,
    mode: ModeName,
    k_frac: float,
    audit_frac: float,
    hard_reject: Sequence[bool],
    scores: Sequence[float | None],
) -> list[tuple[Decision, float, list[str]]]:
    """Return (decision, probability, reasons) per candidate."""
    assert len(hard_reject) == n
    out: list[tuple[Decision, float, list[str]]] = []
    alive_idx = [i for i in range(n) if not hard_reject[i]]
    if mode == "off":
        for i in range(n):
            if hard_reject[i]:
                out.append(("skip", 0.0, ["tier0_hard_reject"]))
            else:
                out.append(("train", 1.0, ["mode_off_train_all"]))
        return out

    # shadow / active share scoring order; shadow still trains all (log only)
    ranked = sorted(
        alive_idx,
        key=lambda i: (scores[i] is not None, scores[i] if scores[i] is not None else float("-inf")),
        reverse=True,
    )
    k = max(1, int(round(len(ranked) * k_frac))) if ranked else 0
    top = set(ranked[:k])
    rest = [i for i in ranked if i not in top]
    n_audit = int(round(len(rest) * audit_frac)) if rest else 0
    audit = set(rest[:n_audit])

    for i in range(n):
        if hard_reject[i]:
            out.append(("skip", 0.0, ["tier0_hard_reject"]))
            continue
        if mode == "shadow":
            out.append(("train", 1.0, ["mode_shadow_train_all", f"would_keep={i in top}"]))
            continue
        # active
        if i in top:
            out.append(("train", k / max(len(ranked), 1), ["active_topk"]))
        elif i in audit:
            out.append(("audit", max(audit_frac, 1e-6), ["active_audit"]))
        else:
            out.append(("skip", 0.0, ["active_filtered"]))
    return out


def mock_seed_labels(seeds: Sequence[int], *, timesteps: int, digest: str) -> list[SeedLabel]:
    labels: list[SeedLabel] = []
    for seed in seeds:
        h = int(hashlib.sha256(f"{digest}|{seed}".encode()).hexdigest()[:8], 16)
        success = (h % 1000) / 1000.0
        collision = ((h // 1000) % 500) / 1000.0
        timeout = max(0.0, 1.0 - success - collision)
        labels.append(
            SeedLabel(
                seed=int(seed),
                success=success,
                collision=collision,
                timeout=timeout,
                return_mean=float((h % 200) - 100) / 10.0,
                eval_episodes=20,
                train_steps=int(timesteps),
                wall_time_s=0.01,
                status="completed",
            )
        )
    return labels


def _best_parents(records: Sequence[StoreRecord], k: int = 2) -> list[StoreRecord]:
    scored: list[tuple[float, StoreRecord]] = []
    for rec in records:
        if not rec.labels:
            continue
        mean_s = sum(lab.success for lab in rec.labels) / len(rec.labels)
        scored.append((mean_s, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]


@dataclass
class GenerationResult:
    generation: int
    records: list[StoreRecord]
    mode: ModeName
    n_train_jobs: int


def run_generation(
    cfg: LoopConfig,
    client: LLMClient,
    *,
    generation: int,
    transitions: Sequence[Mapping[str, Any]],
    prompt_template: str,
    task_text: str,
    parents: Sequence[StoreRecord] = (),
    labeled_so_far: int = 0,
) -> GenerationResult:
    mode: ModeName = cfg.mode
    if mode == "active" and labeled_so_far < cfg.m_min:
        mode = "off"

    responses: list[LLMResponse] = []
    parent_list = list(parents) if parents else [None]
    # Split n across parents (or single gen0 prompt).
    per_parent = max(1, cfg.n_per_generation // max(len(parent_list), 1))
    for parent in parent_list:
        parent_code = parent.candidate.code if parent is not None else ""
        stats = "{}"
        if parent is not None and parent.labels:
            stats = json.dumps(
                {
                    "mean_success": sum(l.success for l in parent.labels) / len(parent.labels),
                    "parent_id": parent.candidate.candidate_id,
                }
            )
        prompt = render_prompt(
            prompt_template,
            task=task_text,
            parent_code=parent_code,
            stats=stats,
        )
        need = per_parent if parent is not None or len(parent_list) == 1 else per_parent
        responses.extend(client.complete(prompt, temperature=cfg.temperature, n=need))
    responses = responses[: cfg.n_per_generation]
    while len(responses) < cfg.n_per_generation:
        responses.extend(
            client.complete(
                render_prompt(prompt_template, task=task_text),
                temperature=cfg.temperature,
                n=1,
            )
        )

    candidates: list[CandidateRecord] = []
    reports = []
    features_list: list[dict[str, float] | None] = []
    hashes: list[str | None] = []
    hard = []
    for i, resp in enumerate(responses[: cfg.n_per_generation]):
        ch = code_hash(resp.text)
        parent = parents[i % len(parents)] if parents else None
        lineage = parent.candidate.lineage_id if parent is not None else f"lin_{cfg.run_id}_{i:03d}"
        cand = CandidateRecord(
            candidate_id=_candidate_id(cfg.run_id, generation, i, ch),
            code=resp.text,
            code_hash=ch,
            host=cfg.host,
            env_id=cfg.env_id,
            lineage_id=lineage,
            parent_id=None if parent is None else parent.candidate.candidate_id,
            prompt_id=cfg.prompt_id,
            generation=generation,
            llm_meta={"model": resp.model, **resp.meta},
        )
        report, feats, fhash, _ = triage_candidate(cand.code, transitions, in_process=True)
        candidates.append(cand)
        reports.append(report)
        features_list.append(feats)
        hashes.append(fhash)
        hard.append(is_hard_reject(report))

    # No surrogate yet in Phase 4 warm-up: scores None → off trains all.
    scores: list[float | None] = [None] * len(candidates)
    decisions = select_decisions(
        len(candidates),
        mode=mode,
        k_frac=cfg.k_frac,
        audit_frac=cfg.audit_frac,
        hard_reject=hard,
        scores=scores,
    )

    records: list[StoreRecord] = []
    jobs_used = 0
    for cand, report, feats, fhash, (decision, prob, reasons) in zip(
        candidates, reports, features_list, hashes, decisions, strict=True
    ):
        labels: list[SeedLabel] = []
        do_label = decision in ("train", "audit")
        if do_label:
            if cfg.max_jobs is not None and jobs_used >= cfg.max_jobs:
                # Record as skipped due to job cap (pilot).
                decision = "skip"
                reasons = list(reasons) + ["max_jobs_cap"]
                do_label = False
            else:
                n_seed_jobs = len(cfg.seeds)
                if cfg.max_jobs is not None:
                    remaining = cfg.max_jobs - jobs_used
                    seeds = list(cfg.seeds)[: max(0, remaining)]
                else:
                    seeds = list(cfg.seeds)
                if cfg.dry_run:
                    labels = mock_seed_labels(
                        seeds, timesteps=cfg.ppo_timesteps, digest=cand.code_hash
                    )
                else:
                    from reward_triage.labeling import label_code_candidate_seeds

                    if cfg.env_config is None or cfg.ppo_config is None or cfg.eval_config is None:
                        raise RuntimeError(
                            "Real labeling requires env_config, ppo_config, and eval_config"
                        )
                    labels = label_code_candidate_seeds(
                        code=cand.code,
                        candidate_id=cand.candidate_id,
                        seeds=seeds,
                        run_root=Path(cfg.experiments_root) / cfg.run_id / "runs",
                        env_config=cfg.env_config,
                        ppo_config=cfg.ppo_config,
                        eval_config=cfg.eval_config,
                        family_id=str(cand.llm_meta.get("family", "local")),
                        enable_early=cfg.enable_early,
                    )
                jobs_used += len(seeds)

        # Merge mean early_* into feature vector when present (early+math+behavioral).
        merged_feats = dict(feats) if feats else None
        if labels and merged_feats is not None:
            early_maps = [lab.early_features for lab in labels if lab.early_features]
            if early_maps:
                keys = set().union(*(m.keys() for m in early_maps))
                for key in keys:
                    vals = [m[key] for m in early_maps if key in m]
                    if vals:
                        merged_feats[key] = float(sum(vals) / len(vals))

        rec = StoreRecord(
            candidate=cand,
            tier0=report,
            features=merged_feats,
            feature_hash=fhash,
            selection=SelectionLog(
                score=None,
                probability=prob,
                decision=decision,  # type: ignore[arg-type]
                mode=mode,
                reasons=reasons,
            ),
            labels=labels,
            audit=(decision == "audit"),
            run_id=cfg.run_id,
            env_config_id=cfg.env_config_id,
        )
        records.append(rec)

    return GenerationResult(
        generation=generation,
        records=records,
        mode=mode,
        n_train_jobs=jobs_used,
    )


def run_loop(
    cfg: LoopConfig,
    client: LLMClient,
    *,
    prompt_gen0: str,
    prompt_mutate: str,
    task_text: str,
) -> dict[str, Any]:
    probe = load_probe_npz(cfg.probe_path)
    transitions = probe.transitions
    store_path = Path(cfg.store_path) if cfg.store_path else Path("experiments") / cfg.run_id / "label_store.jsonl"
    store = LabelStore(store_path)

    all_records: list[StoreRecord] = []
    labeled = 0
    parents: list[StoreRecord] = []
    t0 = time.time()
    job_budget = cfg.max_jobs
    for g in range(cfg.n_generations):
        gen_cfg = replace(cfg, max_jobs=None if job_budget is None else max(0, job_budget))
        template = prompt_gen0 if g == 0 else prompt_mutate
        result = run_generation(
            gen_cfg,
            client,
            generation=g,
            transitions=transitions,
            prompt_template=template,
            task_text=task_text,
            parents=parents,
            labeled_so_far=labeled,
        )
        store.extend(result.records)
        all_records.extend(result.records)
        labeled += sum(1 for r in result.records if r.labels)
        if job_budget is not None:
            job_budget -= result.n_train_jobs
        parents = _best_parents(result.records, k=2)

    errors = validate_store(all_records)
    summary = {
        "run_id": cfg.run_id,
        "env_id": cfg.env_id,
        "dry_run": cfg.dry_run,
        "n_records": len(all_records),
        "n_labeled": sum(1 for r in all_records if r.labels),
        "n_generations": cfg.n_generations,
        "store_path": str(store_path),
        "validation_errors": errors,
        "wall_time_s": round(time.time() - t0, 3),
        "mode": cfg.mode,
    }
    summary_path = store_path.parent / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
