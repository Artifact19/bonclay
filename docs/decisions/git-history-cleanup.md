# Decision: removing old voice files from git history

**Date:** 19 October 2026 · **Status:** prepared, NOT executed · **Owner decides:** Ayush Chougula

Nothing in the real repository was rewritten, pushed, deleted or re-tagged while preparing this. The dry run below
ran in throwaway clones outside the project.

## What is in history

Paths below were found with `git log --all --name-only` and `git rev-list --all --objects`. Names only; contents were
not opened.

| Path | Added in | Removed in | Size | What it is |
|---|---|---|---|---|
| `backend/data/audio/463f8866-5286-4d0b-b778-091f6aaff2c8.pt` | 998dc1e (2026-08-03, "Include all files from backend and frontend") | e28b131 (2026-10-15, "Stop tracking private voice data") | 4,084 B | Voice embedding file named by voice id |
| `backend/data/voices.json` | 998dc1e | e28b131 | 328 B | Local voice list (the app's own `voices.json`) |
| `frontend/.cache/47bf1ba8-bc8c-48ab-9d05-63230057356b/` with 5 `.pt` files (`60cf0a23-...`, `7c02a9d6-...`, `9e2cbe70-...`, `ad56f8cb-...`, `d4ef7392-...`) | 998dc1e | 577b4bc (2026-10-15, "Cleanup ... no private voice data") | four of ~5.5 KB, one of 3.77 MB | Cached tensors keyed by a user id and voice ids (voice-prompt / embedding caches) |
| `backend/question.m4a` | 1c97fda (2026-08-07, "working 80%") | 577b4bc | 83 KB | A recorded spoken question (a voice recording) |

All of them are reachable from `main`, `origin/fix/voice-picker-names`, `origin/fix/windows-nvidia-support` and the
tag `v1.0-demo` (cf06324), because they were added in August and removed on 15 October.

Reviewed and judged not sensitive (leave in history): `frontend/public/landing/voices/*.m4a` and
`frontend/public/samples/languages/*.m4a` (landing-page samples, still tracked at HEAD, presumably rendered with preset
voices; confirm that none of them is a cloned human voice before treating this as closed), `.env.example` files
(templates), `.impeccable/*` design-tool state and screenshots (removed in 577b4bc, no voices), `notebook/Voice_Cloning.ipynb`
(9 KB, tracked despite the `*.ipynb` ignore rule; worth a glance for pasted tokens or paths, not part of this rewrite).
No `.wav`, `.mp3`, `.webm`, real `.env`, `.pem` or key file was ever committed.

## Is a real person's voice involved?

Probably yes, though I cannot prove it without opening the files, which I deliberately did not do. The removing commits
call it "private voice data" ("Stop tracking private voice data (embedding and voice list)"), the `.pt` files are
speaker embeddings or voice-prompt caches that identify a cloned voice, and `question.m4a` is a human recording from
the early voice-cloning prototype. Treat them as biometric-like voice data of whoever recorded them, most likely you or
a teammate. The history has other authors (Arnav Bule, Atharva1399), so ask them whether any of the material is theirs.
If it belongs to someone who has not consented to public hosting, that person has a withdrawal right (see
Consequences).

## Current state at HEAD

- No voice, audio-embedding, `.pt`, `.env` or key file is tracked. Tracked audio is only the `.m4a` landing samples.
- `.gitignore` coverage: `.env` is covered by `backend/.gitignore` and `frontend/.gitignore` has `.env.local` (root
  `.env.example` files are intentionally tracked). `backend/data` is covered per subpath by `backend/.gitignore`
  (`data/audio/`, `data/voices.json`, `data/local_db/`, keys, receipts, integrations, and so on) and `frontend/.cache/`
  by the root `.gitignore`.
- Gaps: there is no blanket `backend/data/` rule, so new files written there show up as untracked (today
  `backend/data/voice_withdrawals.jsonl` is untracked and not ignored: add it, or ignore `data/*.jsonl`). `*.pt` is not
  ignored anywhere (`*.pth` is). `*.m4a`, `*.wav` outside `examples/` and `*.webm` of user recordings are only partly covered.
  Suggested lines for the root `.gitignore`: `*.pt`, `backend/data/*` with `!backend/data/.gitkeep` if you want a
  folder, and `video/` if the untracked `video/` working folder should never be committed. I did not edit `backend/`.
- The only references to the removed paths in the checkout are ignore rules and one prose mention of
  `backend/data/voices.json` in `docs/planning/BONCLAY_ENTIRE_SUMMARY.md`. Nothing in code, tests or build config reads them.

## Dry run (tested in throwaway clones)

Environment: Python 3.12 venv with `pip install git-filter-repo`, scratch clones outside the project.

```bash
git clone --mirror --no-local <repo> mirror.git && cd mirror.git
cat > paths.txt <<'P'
backend/data/voices.json
backend/data/audio/463f8866-5286-4d0b-b778-091f6aaff2c8.pt
backend/question.m4a
frontend/.cache/
P
git filter-repo --invert-paths --paths-from-file paths.txt --force
git reflog expire --expire=now --all && git gc --prune=now
```

Results:

- 72 commits parsed; 71 remain on `main` (the commit e28b131 became empty and was dropped; every other commit hash
  changed from the first commit that touched the paths onward).
- Repository size 31 MB before and 27 MB after (pack 30.5 MB to 27.0 MB; the removed 3.77 MB `.pt` and friends account
  for the difference).
- `git log --all --name-only | grep -E 'backend/data|frontend/.cache|question.m4a'` returned nothing afterwards.
- The tree of the rewritten `main` (452b2cd3...) is identical to the current `main` tree: same 530 files, so no build
  file, source file or test lost anything. No code or test references the removed paths (grep above), so the build and
  tests are unaffected by definition. I did not rerun the whole test suite on the clone.
- In the mirror run: `main` 21672fa to b414bbc, `v1.0-demo` cf06324 to 019bed3, both fix branches got new hashes.

## Real rewrite, exact steps

Do these only when you are sure. Nothing here has been run for real.

```bash
# 0. Pause: tell collaborators (Arnav Bule, Atharva1399) to push nothing; merge or note open work (4 PRs exist).
# 1. Fresh mirror clone, so the working checkout and its untracked files are never touched.
cd ~/Desktop && git clone --mirror https://github.com/Artifact19/bonclay.git bonclay-rewrite.git
cd bonclay-rewrite.git
cp -r . ../bonclay-backup-before-rewrite.git        # keep until you are happy; contains the private files, store safely

# 2. Rewrite
pip install git-filter-repo
cat > ../paths.txt <<'P'
backend/data/voices.json
backend/data/audio/463f8866-5286-4d0b-b778-091f6aaff2c8.pt
backend/question.m4a
frontend/.cache/
P
git filter-repo --invert-paths --paths-from-file ../paths.txt --force
git log --all --name-only --format='' | grep -E 'backend/data|frontend/.cache|question.m4a'   # must print nothing
git rev-parse main^{tree}                                  # must equal the tree of today's main

# 3. Re-add the remote (filter-repo removes it) and force-push branches and tags
git remote add origin https://github.com/Artifact19/bonclay.git
git push --force --mirror origin    # pushes branches and tags; GitHub refuses refs/pull/* , that is expected

# 4. Release tag: v1.0-demo is rewritten by filter-repo (cf06324 -> new hash) and force-pushed by step 3.
#    Re-create the GitHub Release on the new tag if it points at the old commit:
#    gh release view v1.0-demo ; (edit or recreate) gh release create v1.0-demo --target main ...

# 5. Your working checkout: do NOT pull. Keep your untracked files, then re-clone and copy them in.
cd ~/Desktop/Ayush/Projects && mv BonClay BonClay-old && git clone https://github.com/Artifact19/bonclay.git BonClay
cp -R BonClay-old/backend/data BonClay/backend/data     # local-only state (voices, keys, receipts) and video/, .env files
cp -R BonClay-old/video BonClay/video
cp BonClay-old/backend/.env BonClay/backend/.env 2>/dev/null; cp BonClay-old/frontend/.env.local BonClay/frontend/ 2>/dev/null
# 6. Ask GitHub Support (support.github.com, "remove cached views / sensitive data") to run garbage collection and
#    purge refs/pull/*, cached views and forks' references. Give them the four old commit hashes' paths.
```

## Consequences

- Every commit hash from 998dc1e (August) onward changes. Links, issue and PR references to old hashes break.
- The `v1.0-demo` tag must point at the new commit; the tag is rewritten, but any GitHub Release or notes quoting
  cf06324 should be updated.
- Every collaborator must re-clone (or `git fetch` and `git reset --hard origin/main` on a clean checkout). Anyone who
  merges old history back re-introduces the files. Local branches based on old commits must be rebased with
  `git rebase --onto`.
- The four open PR refs (`refs/pull/1-4/head`) cannot be rewritten by push; GitHub keeps them, and they still contain the
  files, as do forks, clones, and cached views. Until GitHub Support purges them, treat the voice data as exposed
  (public repo, so anyone could already have copied it since August).
- Consequently, deleting from history reduces exposure but cannot recall anything. If a real person's voice is in those
  files and they did not consent to public hosting, honour any withdrawal of consent now: tell them what was exposed
  and for how long (3 August to 15 October in the tree, still in history since), delete their voice in the app, and
  record it (the app already logs withdrawals to `backend/data/voice_withdrawals.jsonl`). A rewrite does not undo that
  exposure. Secrets: none found (no real `.env` or keys), so no rotation is required, but if the notebook contains a token, rotate it.
- If you decide not to rewrite, the repo stays valid and the files remain publicly reachable; the cost is the exposure
  above, the benefit is no disruption of hashes, the tag, PRs and collaborators.
