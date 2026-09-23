# Getting this work onto a GPU

Three items in `docs/next.md` are blocked on hardware that does not exist in
this session, and one more keeps failing for a related reason. This is what
the options actually are, what each unblocks, and what it costs.

## What is blocked, and what it is worth

| Item | Needs | Why it matters |
| --- | --- | --- |
| **A.3** Replace the spike with a real backbone | A GPU, hours | The single largest expected accuracy gain, and the last standing explanation for `size` |
| **B.1** The L4 burn-in for `$/MTok` | An L4, ~1 hour | The most load-bearing unmeasured number in the project; the economic case is arithmetic over it |
| **C.1** Publish weights | A.3 | Nothing worth publishing until the model is not a spike |
| **A.2** HelpSteer2 | Not a GPU — *durability* | Three attempts lost; see below |

A.3 and B.1 are the real ones. The rest of the plan is done or does not need
hardware.

## The constraint that is not about speed

Cloud sessions are reclaimed after a period of inactivity, and the
documentation is explicit about what that costs:

> Background work that was still running when the VM was reclaimed, such as
> subagents and shell commands, isn't restored.

That is all three HelpSteer2 losses. The first was an out-of-memory kill, the
second was killed deliberately when it projected to 45 hours, and the third
was a VM reclamation an hour and a half in. **A training run longer than a
session's idle window cannot finish here**, whatever hardware it runs on, so
any answer that keeps the training inside this VM is a workaround rather than
a fix.

That points at the same place a GPU does: run the work somewhere it survives.

## Option 1 — Self-hosted environment on your own GPU box

**Best fit if you have or can rent a machine with a GPU.** Claude Code cloud
sessions can be routed to runners you operate, and the session process then
executes on your host — so its GPU is simply the session's GPU, with no
remote-job plumbing at all. `nvidia-smi` would work, `torch.cuda.is_available()`
would be True, and everything in this repository would run unchanged.

What it needs:

- **Team or Enterprise plan.** It is a public beta and off by default; an
  Owner turns on **Allow self-hosted environments** on the Cloud environments
  admin page. Not available with Zero Data Retention.
- **A host you run**, with the runner process on it. You build and maintain
  the runner image, which is where CUDA and a GPU build of torch would go.
- **Outbound HTTPS only.** The runner polls `api.anthropic.com`; nothing
  connects inward.

Billing is unchanged — sessions consume the same Claude Code usage — so the
only new cost is the machine.

The catch is operational ownership: you maintain the image and the fleet. For
a single GPU box that is small; for a fleet it is a real commitment.

## Option 2 — Keep this environment, push jobs to a rented GPU

**Best fit if you want no new infrastructure to own.** This session already
reaches arbitrary HTTPS hosts through the egress proxy — that is how
Banking77 and HelpSteer2 were downloaded — so a GPU service with an HTTPS API
is reachable from here. I would send code and data out, run training there,
and pull the reports back.

What it needs:

- **An API credential**, added to the cloud environment rather than pasted
  into the repository. Anthropic-hosted environments keep those outside the
  sandbox and attach them after requests leave.
- **The provider's domain allowlisted** in the environment's network access.

The fit matters more than the price here. **Modal** is the closest match: it
is a Python SDK over HTTPS, you declare a function with a GPU attached, and
it ships your code and returns results — no SSH, which this sandbox does not
have. **RunPod** serverless is comparable. Providers whose model is "rent a
box and SSH in" — Lambda, Vast — are a poor fit for exactly that reason.

This also fixes the durability problem as a side effect: the remote job keeps
running when this VM is reclaimed, and I collect the result on the next turn.

## Option 3 — You run it, I prepare it

**Zero setup.** I write the training script and the exact command; you run it
on any machine with a GPU and commit the reports back. Everything in this
repository is already a CLI: `scripts/train_corpus.py`, `scripts/seed_sweep.py`
and `trigon train` all take flags and write reports.

The cost is a slow loop — every iteration goes through you — which is fine for
B.1 (one burn-in, one number) and poor for A.3 (an experiment with several
rounds).

## Reachability, checked rather than assumed

Tested from this session on 2026-09-22, since the whole question turns on what
the egress proxy allows:

| Host | Response |
| --- | --- |
| `api.modal.com` | **200** |
| `api.replicate.com` | 200 |
| `rest.runpod.io` | 301 |
| `cloud.lambdalabs.com` | 301 |
| `api.together.xyz` | 307 |

`pip install modal` also works — the SDK is installed in this session's
virtualenv at 1.5.5. So the HTTPS route is open and the client runs here.

## What I would actually suggest

**Modal, for everything except B.1.** Five reasons, in the order they matter:

1. **It survives the thing that has killed three runs.** The job executes on
   Modal's infrastructure, so this VM being reclaimed costs nothing — I
   collect the result on the next turn. Self-hosting does *not* automatically
   fix this: a self-hosted runner still releases its session, and background
   tasks get about sixty seconds of grace, so a long training run there needs
   the same detach-and-collect pattern anyway.
2. **No SSH, which this sandbox does not have.** Modal ships code over HTTPS
   and returns results the same way. That is why Lambda and Vast, whose model
   is "rent a box and log in", are the wrong shape here regardless of price.
3. **I can iterate without you in the loop.** A.3 is an experiment — several
   configurations, four seeds each, read the spread, adjust — not one command.
   A route that round-trips through a human each cycle turns a day into a
   week.
4. **Setup is about ten minutes**, against building and maintaining a CUDA
   runner image and keeping a fleet alive.
5. **It is already reachable**, verified above.

### What you would need to do

1. Create a Modal account at modal.com. The free tier carries credits that
   likely cover B.1 outright.
2. Run `modal token new` locally; it prints a token id and secret.
3. Add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` as **cloud environment
   variables**, not repository secrets — Anthropic-hosted environments keep
   those outside the sandbox and attach them after requests leave.
4. Nothing else. `modal.com` already resolves through the proxy.

Then I write the Modal app, push the corpus and the training code to it, and
drive A.3 and A.2-at-a-real-size from here.

### Running it

`scripts/modal_train.py` is written and waiting on a token.

```bash
pip install -e ".[gpu]"                 # the launcher only; the job builds its own image
modal run scripts/modal_train.py --corpus helpsteer2 --n 12000 --epochs 6
modal run scripts/modal_train.py --corpus banking77 --gpu L4 --seeds 0,1,2,3
```

Each seed is its own container, so four seeds cost the wall clock of one
rather than four-on-four-cores — which is why this does not reuse
`scripts/seed_sweep.py`, whose parallelism is local processes. The reports
come back as return values and land in `reports/<corpus>/`.

Three things it records rather than assumes, in `modal-run.json` beside the
reports: the **git SHA** the image was built from, the **GPU it actually
got** (not the one requested), and the wall clock. A number from hardware
nobody can name, at a commit nobody can identify, is a claim rather than a
result — and that is the whole difference this project trades on.

The corpus is downloaded inside the job rather than shipped from here: it is
someone else's data under a licence that governs redistribution, and
`trigon.evals.corpora` already fetches it to an ignored cache with the
attribution attached.

**Credentials reach a session through the cloud environment, not the
repository.** Add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` as environment
variables on the environment; they are kept outside the sandbox and attached
after requests leave. They apply to **new** sessions, so the session that
sets them will not see them.

### Where the other two still win

**B.1, the L4 burn-in, does not need any of this.** It is one measurement on
named hardware: rent an L4 for an hour, run one command I will write, commit
the report. Doing it through Modal would work and would measure *Modal's*
L4 under *Modal's* container, which is a fine number and a less direct one
than the burn-in asks for. If you only ever do one of these, do this one.

**Self-hosting wins if you are on Team or Enterprise, already run a GPU box,
and want checkouts and artifacts to stay inside your network.** Then the
session's GPU is simply the GPU and there is no remote-job plumbing at all.
That is a compliance answer more than a convenience one.



**For B.1, option 3.** It is one measurement on named hardware. Rent an L4 for
an hour, run one command, commit the report. It closes the most load-bearing
unmeasured number in the project for about a dollar of compute.

**For A.3, option 1 or 2.** Prefix-LM conversion of a 0.5–1.5B base is an
experiment, not a run: several configurations, four seeds each, read the
spread, adjust. That loop wants the GPU attached to the session rather than a
round trip through a human. Option 1 if you are on Team or Enterprise and
comfortable running a host; option 2 otherwise.

Rough size: at the ~$0.80/hour an L4 costs in `docs/roadmap.md`, a backbone
conversion plus a four-seed certification is tens of GPU-hours. **Order
$50 of compute, not thousands.** The expensive resource here has never been
the hardware.

## What to do if none of this happens

The plan stays honest and stops at the line. A.3, B.1 and C.1 stay open with
*checked* blockers rather than assumed ones, and everything that does not need
hardware is finished. `reports/banking77/` is a certified result on real data,
and the pipeline it exercises does not change when the backbone does.
