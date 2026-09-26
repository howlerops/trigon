// Runs the generated TypeScript client (as JS, types erased) against a live
// gateway. A client that only typechecks proves nothing about the server.
import { TrigonClient, TrigonError, choice, score, noul, CONTRACT_VERSION }
  from "./src/generated.ts";

const base = process.argv[2];
const client = new TrigonClient(base);

const health = await client.healthz();
console.log("contract", CONTRACT_VERSION, "| healthz", health.status,
            "| calibrated", health.calibrated, "| trained", health.trained);
console.log("models  ", JSON.stringify((await client.models()).data));

const response = await client.decide(
  "my card was declined at the till and I still got charged",
  {
    route: choice("Which queue?", [
      { name: "billing", criteria: "a charge, refund or card payment" },
      { name: "shipping", criteria: "a parcel or delivery" },
    ]),
    severity: score("How severe?", [
      { name: "low", value: 1.0 }, { name: "high", value: 5.0 },
    ]),
    urgent: noul("Needs a human within the hour?"),
  },
);
const route = response.answers.route;
console.log("model   ", response.model, "| tier", response.tier);
console.log("route   ", route.selected, JSON.stringify(route.probabilities),
            "conf", route.confidence.toFixed(3));
console.log("severity", response.answers.severity.score);
console.log("urgent  ", response.answers.urgent.probability.toFixed(3),
            "| confidence field present:", "confidence" in response.answers.urgent);
console.log("usage   ", JSON.stringify(response.usage));

try {
  await client.decide("x", { q: choice("pick", ["only"]) });
  console.log("ERROR: a single-option Choice was accepted");
  process.exit(1);
} catch (e) {
  if (!(e instanceof TrigonError)) throw e;
  console.log("bad request ->", e.status, "(TrigonError)");
}
