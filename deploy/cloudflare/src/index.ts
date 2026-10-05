/**
 * The trigon front door: authenticate the caller, forward decisions to the
 * GPU gateway, and serve the open model files.
 *
 * The caller holds one key (CLIENT_KEYS). The Modal proxy token and the
 * gateway's own key live only here, so a caller can neither reach the GPU
 * around this Worker nor learn the credentials that would let them. A
 * request without a valid key is refused here, before anything wakes a GPU.
 */

interface Env {
  ORIGIN: string;
  CLIENT_KEYS: string;
  GATEWAY_KEY: string;
  MODAL_KEY: string;
  MODAL_SECRET: string;
  MODELS?: R2Bucket;
  /**
   * The incumbent's decision path, served at the root so a client that may
   * change only its base URL works unmodified. A deploy-time variable
   * (`wrangler deploy --var COMPAT_PATH:<path>`), never written in this
   * repository: the path names the incumbent (`tests/test_no_incumbent_names.py`).
   */
  COMPAT_PATH?: string;
}

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Authorization, Content-Type",
  "Access-Control-Max-Age": "86400",
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  });
}

/** Constant-time comparison: a key is never matched character by character. */
async function equal(a: string, b: string): Promise<boolean> {
  const encoder = new TextEncoder();
  const [x, y] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(a)),
    crypto.subtle.digest("SHA-256", encoder.encode(b)),
  ]);
  return crypto.subtle.timingSafeEqual(x, y);
}

async function authorised(request: Request, env: Env): Promise<boolean> {
  const header = request.headers.get("Authorization") ?? "";
  if (!header.startsWith("Bearer ")) return false;
  const presented = header.slice("Bearer ".length).trim();
  if (!presented) return false;
  const keys = (env.CLIENT_KEYS ?? "").split(",").map((k) => k.trim()).filter(Boolean);
  let ok = false;
  for (const key of keys) ok = (await equal(presented, key)) || ok; // no early exit
  return ok;
}

/** How long a request may wait for a GPU replica to come up from zero. */
const COLD_START_BUDGET_MS = 90_000;

async function forward(request: Request, env: Env, path: string): Promise<Response> {
  const url = new URL(path + new URL(request.url).search, env.ORIGIN);
  // Buffered once, so a retry can resend it.
  const body = request.method === "GET" ? undefined : await request.arrayBuffer();
  const init = {
    method: request.method,
    body,
    headers: {
      "Content-Type": request.headers.get("Content-Type") ?? "application/json",
      Authorization: `Bearer ${env.GATEWAY_KEY}`,
      "Modal-Key": env.MODAL_KEY,
      "Modal-Secret": env.MODAL_SECRET,
    },
  };
  // A Modal Server answers 503 while no replica is ready -- from zero, for the
  // tens of seconds a GPU container takes to load. Retried here with backoff,
  // so a caller sees one slow request rather than an error.
  const deadline = Date.now() + COLD_START_BUDGET_MS;
  let delay = 500;
  let upstream = await fetch(url, init);
  while (upstream.status === 503 && Date.now() + delay < deadline) {
    await upstream.body?.cancel();
    await new Promise((resolve) => setTimeout(resolve, delay));
    delay = Math.min(delay * 2, 5_000);
    upstream = await fetch(url, init);
  }
  const headers = new Headers(upstream.headers);
  for (const [k, v] of Object.entries(CORS)) headers.set(k, v);
  return new Response(upstream.body, { status: upstream.status, headers });
}

async function model(path: string, env: Env): Promise<Response> {
  if (!env.MODELS) return json({ error: "model storage is not configured" }, 503);
  const key = decodeURIComponent(path.slice("/models/".length));
  if (!key || key.includes("..")) return json({ error: "not found" }, 404);
  const object = await env.MODELS.get(key);
  if (!object) return json({ error: "not found" }, 404);
  const headers = new Headers(CORS);
  object.writeHttpMetadata(headers);
  headers.set("ETag", object.httpEtag);
  // Published bundles never change under a name; a new model is a new path.
  headers.set("Cache-Control", "public, max-age=86400, immutable");
  return new Response(object.body, { headers });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const { pathname, searchParams } = new URL(request.url);
    if (request.method === "OPTIONS") return new Response(null, { headers: CORS });

    if (request.method === "GET" && pathname.startsWith("/models/")) return model(pathname, env);

    if (pathname === "/healthz") {
      // Answered here unless asked to look further: a health probe should not
      // wake a GPU, and a deep one has to be from a caller allowed to.
      if (searchParams.get("deep") !== "1") return json({ status: "ok", edge: true });
      if (!(await authorised(request, env))) return json({ error: "unauthorized" }, 401);
      return forward(request, env, "/healthz");
    }

    const atRoot = !!env.COMPAT_PATH && pathname === env.COMPAT_PATH;
    const decision =
      request.method === "POST" &&
      (pathname === "/v1/decide" || pathname.startsWith("/compat/") || atRoot);
    if (!decision) return json({ error: "not found" }, 404);
    if (!(await authorised(request, env))) return json({ error: "unauthorized" }, 401);
    // The gateway mounts the incumbent's shapes under /compat.
    return forward(request, env, atRoot ? `/compat${pathname}` : pathname);
  },
} satisfies ExportedHandler<Env>;
