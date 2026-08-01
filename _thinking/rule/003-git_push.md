# 003 — git push 방법 (공유 계정 5090에서)

> 🚫 **이 파일에 실제 토큰을 붙여넣지 마라.**
> 이 절차에서 토큰은 **터미널 비밀번호 프롬프트에만** 입력한다(§3-2). 파일에 적을 일이 없다.
> 한 번이라도 커밋되면 나중에 지워도 **git 히스토리에 영구히 남고**, GitHub 의 secret scanning 이
> push 자체를 거부한다(2026-07-31 실제 발생). 히스토리 재작성은 우리 규칙상 금지다.
>
> 이 경고 때문에 이 문서는 한동안 `.gitignore` 에 있었으나, **실제 토큰이 들어갈 자리가 없음을
> 확인하고 2026-08-02 에 추적으로 전환했다.** 다른 컴퓨터에서 push 할 때 이 절차가 필요하기 때문이다.

> 목적: 이 연구실 5090은 **여러 사람이 같은 OS 계정(`rils`)으로** 쓴다. 그래서 git 자격증명이
> 얽혀 push가 막히기 쉽다. 이 문서는 **dlacksdn 저장소에 안전하게 push하는 검증된 방법**을 남긴다.
> (2026-07-24 실제로 이 방법으로 커밋 2743faa push 성공.)

---

## 1. 왜 그냥 push하면 막히나

- 이 계정은 git 전역 설정이 **`credential.helper=store`** → 공유 파일 `~/.git-credentials`에
  저장된 **다른 사람의 토큰**(공유 계정에 저장돼 있다)을 쓴다.
- 그래서 `git push`하면 그 사람 계정으로 인증 시도 → dlacksdn 저장소엔 쓰기 권한이 없어 **403**:
  ```
  remote: Permission to dlacksdn/2026_Inha_AI_challenge_WM.git denied to <다른-github-계정>.
  ```
- 또 이 저장소는 **HTTPS** remote라, GitHub는 비밀번호가 아니라 **PAT(개인 액세스 토큰)** 를 요구한다.

## 2. 원칙 (공유 머신이라 중요)

- ❌ `git config --global credential.helper store` 로 재로그인 **금지** — 다른 사람 토큰을 덮어쓴다.
- ❌ 토큰을 디스크에 저장 **지양** — 같은 UID를 쓰는 다른 사람이 읽을 수 있다.
- ❌ VS Code의 "GitHub 로그인" 팝업 **쓰지 말 것**(No) — 세션이 계정끼리 꼬인다.
- ✅ **dlacksdn PAT + 저장 안 하는 1회 push** 가 가장 안전하고 깔끔하다.

## 3. 절차

### 3-1. dlacksdn PAT 발급 (한 번)
github.com에 **dlacksdn 계정으로 로그인** →
Settings → (왼쪽 맨 아래) **Developer settings** → **Personal access tokens → Fine-grained tokens** →
**Generate new token**:
- Resource owner: `dlacksdn`
- Repository access: **Only select repositories** → `2026_Inha_AI_challenge_WM`
- Permissions → Repository permissions → **Contents: Read and write** (Metadata는 자동 Read-only, 필수)
- Expiration: 7일 등 짧게
- Generate → `github_pat_...` **복사**(이 화면 벗어나면 다시 못 봄)

### 3-2. 검증된 push 명령 (본인 터미널에서)
```bash
cd ~/dlacksdn/2026_Inha_AI_challenge_WM
# VS Code의 askpass를 꺼서 팝업 대신 터미널에서 직접 입력받게 함
unset GIT_ASKPASS VSCODE_GIT_ASKPASS_NODE VSCODE_GIT_ASKPASS_MAIN
GIT_TERMINAL_PROMPT=1 git -c credential.helper= push origin main
#   Username: dlacksdn
#   Password: github_pat_...  (붙여넣기, 화면엔 안 보임 / 터미널 붙여넣기는 Ctrl+Shift+V)
```
- `-c credential.helper=` : 공유 `store` 헬퍼를 **이 명령에서만** 끔 → 토큰 어디에도 저장 안 됨.
- `unset ... GIT_ASKPASS` : VS Code GitHub 팝업 우회, 순수 터미널 프롬프트로.

성공 예:
```
To https://github.com/dlacksdn/2026_Inha_AI_challenge_WM.git
   99b75b1..2743faa  main -> main
```

## 4. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `Permission ... denied to <다른-계정>` (403) | 공유된 다른 사람 토큰으로 인증됨 | 위 3-2 명령(dlacksdn PAT 1회) |
| `Password authentication is not supported` / `Invalid username or token` | Password 칸에 **토큰이 아닌 값**(빈값·비번·오타·일부만 붙여넣기) | `github_pat_...` 전체를 다시 붙여넣기 |
| VS Code "finish authorizing... try a different way" 팝업 | VS Code가 git 자격요청을 가로챔 | 팝업 **No**, 3-2의 `unset GIT_ASKPASS...` 로 우회 |

## 5. 참고 (이미 설정된 것)
- 이 저장소에 **로컬 git 신원**이 설정돼 있다(전역 아님 → 공유·타인에 영향 없음):
  `git config --local user.name dlacksdn`, `user.email dlacks3174@gmail.com` (→ `.git/config`).
- 커밋 정책은 [CLAUDE.md](../../CLAUDE.md) "Git 커밋·push 정책" 참고: 분기마다 커밋+push,
  **기존 remote/브랜치만**, force-push 금지, push 전 비밀값·대용량 확인.
