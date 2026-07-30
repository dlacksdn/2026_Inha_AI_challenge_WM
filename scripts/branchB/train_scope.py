"""학습 범위(scope) 정의 — 어떤 파라미터를 학습하고 어떤 것을 얼릴지 한 곳에서 정한다.

주의(005 오진의 원인): UNet **키 이름에 'temporal' 문자열이 없다.**
시간축 레이어는 `TemporalTransformer` / `TemporalConvBlock` 이라는 **모듈 타입**으로만 식별된다.
그래서 이름 매칭이 아니라 타입으로 접두사를 모은 뒤 파라미터를 분류한다.

범위
  full            : 전부 (96GB 전용 — 산술 31.2GB + 활성화)
  action_temporal : 액션분기 + 시간/fps 임베딩 + 시간축 레이어 (38.3%, 산술 21.3GB) ← 32GB용
  action_only     : 액션분기 + 조건 임베딩만 (0.4%)
"""
from __future__ import annotations

SCOPES = {
    "full": ("action", "cond_emb", "temporal", "spatial"),
    "action_temporal": ("action", "cond_emb", "temporal"),
    "action_only": ("action", "cond_emb"),
}


def temporal_prefixes(unet) -> tuple[set, list]:
    """시간축 모듈의 파라미터 이름 접두사 집합과 발견된 타입 목록."""
    names, types = set(), []
    for mod_name, mod in unet.named_modules():
        tn = type(mod).__name__
        if "Temporal" in tn:
            names.add(mod_name)
            types.append(tn)
    return names, sorted(set(types))


def param_group(pname: str, tprefix: set) -> str:
    if pname.startswith("action_embed") or pname.startswith("null_action_emb"):
        return "action"
    if pname.startswith("time_embed") or pname.startswith("fps_embedding"):
        return "cond_emb"
    for t in tprefix:
        if pname.startswith(t + "."):
            return "temporal"
    return "spatial"


def group_stats(unet) -> tuple[dict, list, set]:
    tprefix, types = temporal_prefixes(unet)
    by: dict[str, list] = {}
    for n, p in unet.named_parameters():
        g = param_group(n, tprefix)
        by.setdefault(g, [0, 0])
        by[g][0] += 1
        by[g][1] += p.numel()
    return by, types, tprefix


def apply_scope(unet, scope: str) -> dict:
    """requires_grad 를 설정하고 통계를 돌려준다. (동결 파라미터는 AdamW state 도 만들지 않는다)"""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope: {scope} (가능: {list(SCOPES)})")
    tprefix, types = temporal_prefixes(unet)
    keep = set(SCOPES[scope])
    tr_n = tr_p = fr_n = fr_p = 0
    per_group: dict[str, list] = {}
    for n, p in unet.named_parameters():
        g = param_group(n, tprefix)
        on = g in keep
        p.requires_grad_(on)
        per_group.setdefault(g, [0, 0, 0])          # [키, 파라미터, 학습대상여부]
        per_group[g][0] += 1
        per_group[g][1] += p.numel()
        per_group[g][2] = int(on)
        if on:
            tr_n += 1
            tr_p += p.numel()
        else:
            fr_n += 1
            fr_p += p.numel()
    total = tr_p + fr_p
    return {
        "scope": scope,
        "temporal_module_types": types,
        "trainable_keys": tr_n, "trainable_params_m": round(tr_p / 1e6, 3),
        "frozen_keys": fr_n, "frozen_params_m": round(fr_p / 1e6, 3),
        "trainable_share_pct": round(tr_p / total * 100, 2) if total else 0.0,
        "by_group": {g: {"keys": v[0], "params_m": round(v[1] / 1e6, 3), "trainable": bool(v[2])}
                     for g, v in sorted(per_group.items())},
    }
