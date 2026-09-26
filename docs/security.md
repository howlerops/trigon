# Security and the public repository

What protects the infrastructure this repository drives, and what has to be
true before it is made public. Written 2026-09-25, when the owner decided to
open it and asked for the infrastructure to be secured first.

## What is exposed, and what stops it

| Surface | Exposure once public | What stops misuse |
| --- | --- | --- |
| The served model (`scripts/modal_serve.py`) | Its URL is in `reports/banking77/README.md` | **Modal proxy auth.** A request without a `Modal-Key` / `Modal-Secret` pair is refused with a 401 at Modal's edge in about 0.2 s, **before any container starts**. Verified: three unauthenticated requests and one with a bogus key pair were all refused, and no serving container appeared. Behind it, the gateway's own API key (`TRIGON_API_KEYS`, from the `trigon-serve-auth` Modal Secret) and a per-key rate limit |
| GPU spend on that endpoint | A flood of authenticated traffic | `max_containers=2`: at most two A10Gs at once, whatever arrives; extra requests queue |
| Training and burn-in (`modal_train.py`, `modal_burn_in.py`) | The code only | They run as whoever holds `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`. They expose no web endpoint, and the `trigon-runs` and `trigon-weights` Volumes are private to the workspace |
| CI (`.github/workflows/ci.yml`) | Anyone can open a pull request from a fork | Every job's token is `contents: read`. **A fork's pull request always runs on GitHub's runners, never on the self-hosted one**. The runner is chosen by an expression in the workflow rather than left to a setting, because a self-hosted runner executes whatever a PR contains on our machine. The push-only `reference-run` job can use the self-hosted runner, since only people who can push reach it |
| Pages (`pages.yml`) | The rendered site | Deploys only from the default branch, on push; never from a pull request |
| Releases (`release.yml`) | The weights bundle | Manual only and default branch only. It reads the Modal Volume with the `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` repository secrets, which no pull-request workflow receives, and refuses a bundle whose SHA-256 differs from the committed one |
| Credentials | Anything committed | `tests/test_no_secrets.py` fails the build on anything shaped like a Modal, GitHub, Hugging Face, AWS or bearer credential in a tracked file, and prints only the location. The full history was scanned before the repository went public: nothing |

Two credentials, on purpose. The Modal proxy token decides who may *reach*
the deployment; the trigon API key decides who may *use* it, and carries the
rate limit. Either one leaking alone does not open the endpoint.

## Before flipping the repository to public

Owner actions. None of them can be done from a coding session:

1. **Create a Modal proxy auth token** (Modal dashboard, Settings → Proxy Auth
   Tokens), and give it to whoever calls the endpoint, including the sessions
   that test it, as `MODAL_PROXY_KEY` / `MODAL_PROXY_SECRET`. Until it exists,
   the endpoint answers nobody, which is the safe default. `modal curl` cannot
   substitute for it: it only authenticates Modal's "flash" endpoints.
2. **Set a spending limit on the Modal workspace.** `max_containers` caps the
   serving app. It does not cap a training launch, which anyone holding the
   Modal token can make.
3. **Repository settings**, under Actions → General: require approval for
   workflow runs from all outside contributors. The workflow already keeps
   forks off the self-hosted runner; this also stops them spending minutes
   unreviewed.
4. **Branch protection on the default branch**: pull requests required, and
   CI green before merge. A public repository's default branch is what Pages
   publishes and what `modal_serve.py` is deployed from.
5. **Enable private vulnerability reporting** (Security → Settings) so a
   finding has somewhere to go that is not a public issue.
6. **Set `CI_RUNNER`** (a repository variable) to the self-hosted runner's
   label once the fly.io runner is registered. Until then, every job runs on
   GitHub's runners, which are free on a public repository.

## What a public repository reveals, and that is fine

- The Modal workspace name, in the endpoint URL.
- The commit authors' email addresses.
- How the model was trained, every report, and the release checksums.
- Not included: weights (`*.pt` is ignored, and the adapter is on the Volume),
  corpora (fetched to an ignored cache and never committed), and any secret
  (see above).
