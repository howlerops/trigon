# SDKs

Both clients are **generated** from `spec/openapi.json` by
`scripts/generate_sdk.py`, which is the only definition of the contract. A
hand-written client is a second definition, and two definitions drift.
`tests/test_sdk.py` runs the generator with `--check`, so a contract change
that is not regenerated fails CI in the commit that made it.

Neither client has a runtime dependency. An SDK is the first thing a
prospective user installs, and one that drags in an HTTP stack to call three
endpoints is a worse first impression than forty lines of `urllib` or `fetch`.

| | Path | Runtime | Tested by |
| --- | --- | --- | --- |
| Python | `sdk/python/trigon_client/` | standard library (`urllib`) | the live gateway, in-process |
| TypeScript | `sdk/typescript/src/generated.ts` | `fetch` | `tsc --strict`, then the live gateway from node |

Both are tested against a running server rather than a fixture: a client that
parses a fixture proves nothing about the server it claims to speak to.

## Python

```python
from trigon_client import TrigonClient, choice, noul

client = TrigonClient("http://localhost:8000")
response = client.systemone(
    state="my card was declined at the till and I still got charged",
    questions={
        "route": choice("Which queue?", ["billing", "shipping", "account"]),
        "urgent": noul("Needs a human within the hour?"),
    },
)
response.answers["route"].selected  # "billing"
response.answers["route"].confidence  # a number you can threshold
response.answers["urgent"].probability  # no confidence field, by design
```

## TypeScript

```ts
import { TrigonClient, choice, noul } from "@trigon/client";

const client = new TrigonClient("http://localhost:8000");
const response = await client.systemone(
  "my card was declined at the till and I still got charged",
  {
    route: choice("Which queue?", ["billing", "shipping", "account"]),
    urgent: noul("Needs a human within the hour?"),
  },
);
response.answers.route.selected;      // "billing"
response.answers.route.confidence;    // a number you can threshold
response.answers.urgent.probability;  // no confidence field, by design
```

`npm install && npm run typecheck` in `sdk/typescript`. The generated file is
checked under `strict` with `exactOptionalPropertyTypes` and
`noUncheckedIndexedAccess`, because nobody reads generated code — the compiler
is the review, and it caught a real bug the first time it ran.

## Regenerating

```bash
python scripts/export_openapi.py   # after any change to trigon.types
python scripts/generate_sdk.py     # both clients, from that spec
```
