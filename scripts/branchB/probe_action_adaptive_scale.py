"""009 §10 질문 4 — 액션 조건만으로 표본별 최적 모션량을 예측해 oracle(−2.1%p)에 근접할 수 있나.

배경 (008 §1.5)
  무학습 모션 스윕 24변형에서, 고정 파라미터 최선(motion_b0.10)은 static 대비 −0.00114 (t=−0.64, 유의 X).
  표본별 최적을 **정답을 보고** 고르는 oracle 은 −0.02056. 그 사이 간극이 "표본별 적응"의 몫이다.
  적응의 입력으로 쓸 수 있는 것은 **정답 없이 항상 갖고 있는 것**뿐인데, 그것이 액션(16×6)이다.
  ⇒ 액션에서 뽑은 특징으로 변형을 골라 oracle 의 몇 %를 회수할 수 있는지 정직하게 측정한다.

정직성 장치 (이게 이 실험의 핵심이다)
  1) **LOO-CV**: 표본 i 의 변형 선택에 표본 i 의 점수를 절대 쓰지 않는다(96폴드).
  2) **정규화 강도(alpha)도 폴드 내부 5-fold 로만 고른다**(하이퍼파라미터 누출 차단).
  3) **셔플 대조군**: 특징을 무작위 치환해 같은 파이프라인을 20회 돌린다.
     진짜 신호가 있다면 실측이 셔플 분포 밖에 있어야 한다(008 §2 의 방법론을 그대로 차용).

출력: results/branchB/action_adaptive_scale.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "results/motion_sweep/motion_sweep_report.json"
ACT_DIR = REPO / "artifacts/holdout/actions"
STATS = REPO / "open/data/train/so100_action_statistics.json"
OUT = REPO / "results/branchB/action_adaptive_scale.json"
STATIC = "blend_a1.00"
METRIC = "total_frame_avg"


# ── 데이터 ────────────────────────────────────────────────────────────────────────
def load_scores() -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    rep = json.loads(SWEEP.read_text(encoding="utf-8"))
    variants = list(rep["results"])
    sids = [r["sid"] for r in rep["results"][variants[0]]["rows"]]
    total = np.zeros((len(sids), len(variants)))
    dino = np.zeros_like(total)
    for j, v in enumerate(variants):
        rows = {r["sid"]: r for r in rep["results"][v]["rows"]}
        for i, s in enumerate(sids):
            total[i, j] = rows[s][METRIC]
            dino[i, j] = rows[s]["dino_frame_avg"]
    return sids, variants, total, dino


def load_features(sids: list[str]) -> tuple[np.ndarray, list[str]]:
    st = json.loads(STATS.read_text(encoding="utf-8"))
    mean = np.asarray(st["mean"], dtype=np.float64)
    std = np.asarray(st["std"], dtype=np.float64)
    feats, names = [], None
    for s in sids:
        a = np.load(ACT_DIR / f"{s}.npy").astype(np.float64)      # (16,6) 원 단위
        z = (a - mean) / np.maximum(std, 1e-8)                     # 채점기와 동일한 z 공간
        d = np.diff(z, axis=0)                                     # (15,6) 스텝 변화
        step = np.linalg.norm(d, axis=1)                            # (15,)
        f = {
            "disp_end": float(np.linalg.norm(z[-1] - z[0])),        # 시작→끝 이동량
            "path_len": float(step.sum()),                          # 경로 길이
            "step_max": float(step.max()),
            "step_std": float(step.std()),
            "dim_range_max": float((z.max(0) - z.min(0)).max()),
            "grip_range": float(z[:, 5].max() - z[:, 5].min()),
        }
        for k in range(6):
            f[f"path_d{k}"] = float(np.abs(d[:, k]).sum())
        if names is None:
            names = list(f)
        feats.append([f[k] for k in names])
    return np.asarray(feats), names


# ── 회귀 (numpy 닫힌형 ridge) ──────────────────────────────────────────────────────
def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    d = Xs.shape[1]
    P = np.eye(d)
    P[-1, -1] = 0.0                                  # 절편은 벌하지 않는다
    w = np.linalg.solve(Xs.T @ Xs + alpha * P, Xs.T @ y)
    return (mu, sd, w)


def ridge_pred(model, X: np.ndarray) -> np.ndarray:
    mu, sd, w = model
    return np.hstack([(X - mu) / sd, np.ones((len(X), 1))]) @ w


ALPHAS = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def pick_alpha(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> float:
    """학습 폴드 내부 5-fold 로만 alpha 를 고른다 (Y: (n, k) 다중 타깃)."""
    n = len(X)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)
    err = []
    for a in ALPHAS:
        e = 0.0
        for f in folds:
            tr = np.setdiff1d(idx, f)
            m = ridge_fit(X[tr], Y[tr], a)
            e += float(((ridge_pred(m, X[f]) - Y[f]) ** 2).sum())
        err.append(e)
    return ALPHAS[int(np.argmin(err))]


# ── 정책 ─────────────────────────────────────────────────────────────────────────
def policy_regress_argmin(X: np.ndarray, total: np.ndarray, cand: np.ndarray, seed: int = 0) -> np.ndarray:
    """LOO: 후보 변형별 TOTAL 을 회귀로 예측하고 예측 최소 변형을 고른다 → 선택된 열 index."""
    n = len(X)
    choice = np.zeros(n, dtype=int)
    for i in range(n):
        tr = np.delete(np.arange(n), i)
        rng = np.random.default_rng(seed * 1000 + i)
        Y = total[np.ix_(tr, cand)]
        a = pick_alpha(X[tr], Y, rng)
        m = ridge_fit(X[tr], Y, a)
        pred = ridge_pred(m, X[i:i + 1])[0]
        choice[i] = cand[int(np.argmin(pred))]
    return choice


def summarize(name: str, chosen: np.ndarray, total: np.ndarray, dino: np.ndarray,
              base_col: int) -> dict:
    n = len(chosen)
    tv = total[np.arange(n), chosen]
    dv = dino[np.arange(n), chosen]
    d = tv - total[:, base_col]
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(n))) if d.std() > 0 else 0.0
    return {"policy": name, "total": float(tv.mean()), "dino": float(dv.mean()),
            "delta_vs_static": float(d.mean()), "t_vs_static": round(t, 2),
            "win_rate_vs_static": float((d < 0).mean()),
            "n_distinct_choices": int(len(set(chosen.tolist())))}


def main() -> None:
    sids, variants, total, dino = load_scores()
    X, names = load_features(sids)
    n, V = total.shape
    vi = {v: j for j, v in enumerate(variants)}
    static_j = vi[STATIC]
    fixed_j = int(np.argmin(total.mean(0)))
    motion_cols = np.array([j for j, v in enumerate(variants) if v.startswith("motion_b")])
    all_cols = np.arange(V)

    rep: dict = {"n": n, "variants": variants, "features": names,
                 "static": STATIC, "fixed_best": variants[fixed_j]}

    # 기준선
    rows = [
        summarize("static", np.full(n, static_j), total, dino, static_j),
        summarize(f"fixed_best({variants[fixed_j]})", np.full(n, fixed_j), total, dino, static_j),
        summarize("oracle_all24", total.argmin(1), total, dino, static_j),
        summarize("oracle_motion_family", motion_cols[total[:, motion_cols].argmin(1)],
                  total, dino, static_j),
    ]

    # 예측 정책
    ch_all = policy_regress_argmin(X, total, all_cols)
    rows.append(summarize("pred_regress_argmin(all24)", ch_all, total, dino, static_j))
    ch_mot = policy_regress_argmin(X, total, motion_cols)
    rows.append(summarize("pred_regress_argmin(motion_family)", ch_mot, total, dino, static_j))
    gate = np.array([static_j, fixed_j])
    ch_gate = policy_regress_argmin(X, total, gate)
    rows.append(summarize("pred_gate(static_vs_fixed_best)", ch_gate, total, dino, static_j))

    # 셔플 대조군 (특징만 섞는다 → 신호가 있으면 실측이 이 분포 밖에 있어야 한다)
    sh = {"all24": [], "motion_family": []}
    for r in range(20):
        perm = np.random.default_rng(100 + r).permutation(n)
        sh["all24"].append(summarize("sh", policy_regress_argmin(X[perm], total, all_cols),
                                     total, dino, static_j)["total"])
        sh["motion_family"].append(summarize("sh", policy_regress_argmin(X[perm], total, motion_cols),
                                             total, dino, static_j)["total"])
    rep["shuffle_control"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                 "min": float(np.min(v)), "max": float(np.max(v)), "runs": len(v)}
                             for k, v in sh.items()}

    # 상관: 표본별 최적 b 와 액션 특징
    b_of = {j: float(variants[j].replace("motion_b", "")) for j in motion_cols}
    best_b = np.array([b_of[j] for j in motion_cols[total[:, motion_cols].argmin(1)]])
    rep["corr_bestb_vs_features"] = {
        nm: round(float(np.corrcoef(X[:, k], best_b)[0, 1]), 4) for k, nm in enumerate(names)}
    rep["best_b_distribution"] = {str(b): int((best_b == b).sum()) for b in sorted(set(best_b.tolist()))}
    rep["table"] = rows

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    w = max(len(r["policy"]) for r in rows) + 2
    print("=" * (w + 62))
    print("질문4 — 액션으로 표본별 최적 모션량을 고를 수 있나 (n=96, LOO-CV, 0에 가까울수록 좋음)")
    print("=" * (w + 62))
    print(f"{'정책':<{w}}{'TOTAL':>10}{'DINO':>9}{'Δ vs static':>13}{'t':>7}{'승률':>7}{'선택수':>7}")
    print("-" * (w + 62))
    for r in rows:
        print(f"{r['policy']:<{w}}{r['total']:>10.5f}{r['dino']:>9.5f}{r['delta_vs_static']:>13.5f}"
              f"{r['t_vs_static']:>7.2f}{r['win_rate_vs_static']*100:>6.0f}%{r['n_distinct_choices']:>7}")
    print("-" * (w + 62))
    for k, v in rep["shuffle_control"].items():
        print(f"셔플 대조({k}): TOTAL {v['mean']:.5f} ± {v['sd']:.5f}  (범위 {v['min']:.5f}~{v['max']:.5f}, {v['runs']}회)")
    print("\n표본별 최적 b 분포:", rep["best_b_distribution"])
    top = sorted(rep["corr_bestb_vs_features"].items(), key=lambda kv: -abs(kv[1]))[:5]
    print("최적 b 와 액션 특징의 상관(상위 5):", ", ".join(f"{k} {v:+.3f}" for k, v in top))
    print(f"\n[probe] 리포트: {OUT}")


if __name__ == "__main__":
    main()
