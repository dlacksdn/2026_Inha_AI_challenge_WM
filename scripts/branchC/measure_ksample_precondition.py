"""평균내기의 전제조건을 n=96 으로 확정한다 — 채점 모델 없이 픽셀 통계만으로.

물음: 우리 모델의 오차는 **분산**인가 **편향**인가?
  pred_k = 정답 + 편향 + 잡음_k  라 두면
    한 장의 오차   = |편향 + 잡음|
    K장 평균의 오차 = |편향 + 잡음/K|     ← 잡음만 줄어든다
  ⇒ K장 평균의 오차가 한 장보다 확실히 작으면 **분산이 지배한다** → 평균이 듣는다.

⚠ 픽셀 L1 은 채점 점수가 아니다. 이건 전제조건 확인이고, 점수 확정은 별도 체인이 낸다.
"""
import sys, json, numpy as np, torch
from pathlib import Path
R = Path('/home/rils/dlacksdn/2026_Inha_AI_challenge_WM')
sys.path.insert(0, str(R/'src')); sys.path.insert(0, str(R/'open/submission_kit'))
from wm_eval import scoring as S
import feature_csv_utils as F
sc = S.LocalScorer.__new__(S.LocalScorer)
sc.F = F; sc.TARGET_H, sc.TARGET_W, sc.TEMPORAL, sc.PAD = 320, 512, 16, True

roots = [R/f'artifacts/branchC/ksample/seed{s}' for s in (0,1,2)]
sids = [m["sid"] for m in json.loads((R/'artifacts/holdout/manifest.json').read_text())["samples"]]
rows = []
for i, sid in enumerate(sids):
    gt = sc._load_video(R/'artifacts/holdout/gt_videos', sid).float()
    st = sc._load_video(R/'artifacts/branchB/m0_step1000_b4/static_preds', sid).float()
    P  = torch.stack([sc._load_video(r, sid).float() for r in roots])
    avg = P.mean(0)
    rows.append(dict(
        sid=sid,
        e1m = float((P-gt).abs().mean()),
        e3  = float((avg-gt).abs().mean()),
        est = float((st-gt).abs().mean()),
        sd  = float(P.std(0, unbiased=True).mean()),
        chg = float((gt-st).abs().mean()),
    ))
    if (i+1) % 8 == 0: print(f"  {i+1}/{len(sids)}", flush=True)

M = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "sid"}
W=78; print("\n"+"="*W)
print(f"평균내기의 전제조건 (n={len(sids)}, 픽셀 L1, 255 스케일, K=3)")
print("="*W)
print(f"  정지영상의 오차            {M['est']:.3f}   ← 넘어야 할 선")
print(f"  샘플 한 장의 오차          {M['e1m']:.3f}")
print(f"  **세 장 평균의 오차**      {M['e3']:.3f}   ({(M['e3']/M['e1m']-1)*100:+.1f}%)")
print()
print(f"  샘플 간 흔들림(표준편차)   {M['sd']:.3f}")
print(f"  맞혀야 할 변화량           {M['chg']:.3f}")
print(f"  흔들림/변화량              {M['sd']/M['chg']:.3f}")
d = np.array([r['e3']-r['e1m'] for r in rows]); se = d.std(ddof=1)/np.sqrt(len(d))
print(f"\n  짝지은 비교 (평균 − 한장): Δ={d.mean():+.4f}  SE={se:.4f}  t={d.mean()/se:+.2f}  "
      f"승 {int((d<0).sum())}/{len(d)}")
print("-"*W)
if d.mean() < 0 and d.mean()/se < -2:
    print("  → **평균이 픽셀 오차를 유의하게 줄인다. 분산이 지배한다.**")
    print("     다만 픽셀 오차 ≠ 채점 점수다. 채점 기준 확정은 K샘플 체인이 낸다.")
else:
    print("  → 평균이 픽셀 오차를 못 줄인다. 편향이 지배할 가능성이 크다.")
print(f"  → 정지영상 대비: 평균 {M['e3']:.3f} vs 정지 {M['est']:.3f} "
      f"({'낫다' if M['e3']<M['est'] else '나쁘다'})")
json.dump({"n": len(rows), "means": M, "rows": rows},
          open(R/'results/branchC/ksample_precondition.json','w'),
          ensure_ascii=False, indent=2)
print(f"\n저장: results/branchC/ksample_precondition.json")
