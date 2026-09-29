# Quickstart: TypeScript, Go, and anything that speaks HTTP

Trigon is an HTTP service. Every client below calls the same three endpoints
— `POST /v1/decide`, `GET /v1/models`, `GET /healthz` — defined once in
`spec/openapi.json`, and the TypeScript, Go and Python clients are generated
from that file, so none can drift from it.

## 1. Run a gateway

**In a container** (`ghcr.io/howlerops/trigon`, CPU):

```bash
# The lexical floor: no weights, answers instantly, useful for wiring code up.
docker run -p 8000:8000 ghcr.io/howlerops/trigon

# The certified Banking77 model (v2, robust to injected instructions). The
# bundle is fetched and refused unless its SHA-256 matches.
docker run -p 8000:8000 -v trigon-cache:/cache \
  -e TRIGON_BUNDLE_URL=https://github.com/howlerops/trigon/releases/download/banking77-qwen15b-v2/banking77-qwen15b-v2.tar.gz \
  -e TRIGON_BUNDLE_SHA256=37f3b976d5c7ea655a8bbb65044a3250993cdd906cc008946b093a1749966bee \
  -e TRIGON_API_KEYS=change-me \
  ghcr.io/howlerops/trigon
```

The first start fetches the pinned Qwen2.5-1.5B backbone (3 GB) into the
`trigon-cache` volume; later starts reuse it. On a CPU a request takes about a
second; `/healthz` says whether the calibrators loaded. Every `TRIGON_*`
variable `trigon serve` reads works here (`src/trigon/server/config.py`), and
you can mount an adapter instead of fetching one:
`-v ./adapter.pt:/model/adapter.pt -e TRIGON_BACKEND=torch -e TRIGON_WEIGHTS=/model/adapter.pt`.

**The hosted deployment** on Modal takes two credentials: a Modal proxy token
(`Modal-Key` and `Modal-Secret` headers) and a trigon API key
(`Authorization: Bearer ...`). Every client below accepts extra headers.

## 2. Call it

### curl

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "state": "my card was declined at the till and I still got charged",
  "questions": {
    "route":  {"type": "choice", "instructions": "Which queue?",
               "options": [{"name": "billing"}, {"name": "shipping"}]},
    "urgent": {"type": "noul", "instructions": "Needs a human within the hour?"}
  }
}'
```

### TypeScript — `@howlerops/trigon-client`

Published to GitHub Packages. Point the `@howlerops` scope at it once, with a
GitHub token that has `read:packages` (GitHub Packages asks for one even for
public packages):

```bash
echo "@howlerops:registry=https://npm.pkg.github.com" >> .npmrc
echo "//npm.pkg.github.com/:_authToken=\${GITHUB_TOKEN}" >> .npmrc
npm install @howlerops/trigon-client
```

```ts
import { TrigonClient, choice, noul, TrigonError } from "@howlerops/trigon-client";

const client = new TrigonClient({
  baseUrl: "http://localhost:8000",
  headers: { Authorization: `Bearer ${process.env.TRIGON_API_KEY}` },
});
const r = await client.decide("my card was declined at the till and I still got charged", {
  route: choice("Which queue?", ["billing", "shipping", "account"]),
  urgent: noul("Needs a human within the hour?"),
});
r.answers.route.selected;      // "billing"
r.answers.route.confidence;    // a number you can threshold
r.answers.urgent.probability;  // no confidence field, by design
```

ESM, typed, no runtime dependencies, Node 18 or later (it uses `fetch`).
A non-2xx response throws `TrigonError` with `status` and `payload`.

### Go — `github.com/howlerops/trigon/sdk/go`

```bash
go get github.com/howlerops/trigon/sdk/go@latest
```

```go
import trigon "github.com/howlerops/trigon/sdk/go"

client := trigon.NewClient("http://localhost:8000")
client.Headers["Authorization"] = "Bearer " + os.Getenv("TRIGON_API_KEY")

resp, err := client.Decide(ctx,
    "my card was declined at the till and I still got charged",
    map[string]trigon.Question{
        "route":  trigon.Choice("Which queue?", "billing", "shipping", "account"),
        "urgent": trigon.Noul("Needs a human within the hour?"),
    }, nil)
if err != nil { /* *trigon.Error carries Status and Payload */ }
resp.Answers["route"].Choice.Selected     // "billing"
resp.Answers["urgent"].Noul.Probability  // no confidence field, by design
```

Standard library only. Answers are a tagged union on `Type`, with exactly one
of `Choice`, `Score` or `Noul` set.

### Already on the incumbent's API?

Change the base URL to a trigon gateway's `/compat` path and keep your client:
`docs/compat.md`.

## What each client is tested against

Each one runs against a live gateway in CI, not a fixture: the TypeScript
client from node, the Go client with `go run`, the Python client in-process
(`tests/test_sdk.py`). The image is started and asked for `/healthz` by the
workflow that publishes it (`.github/workflows/publish.yml`).
