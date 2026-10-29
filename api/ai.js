/* Server-side AI proxy for the whiteboard.
   Provider keys live in Vercel env vars (GROQ_KEY / OPENROUTER_KEY) and never
   reach the browser. The client sends { provider, messages, jsonMode, maxTokens, stream }
   and receives the provider's OpenAI-shaped response (including SSE when streaming). */

export const config = { runtime: 'edge' };

const PROVIDERS = {
  groq: {
    url: 'https://api.groq.com/openai/v1/chat/completions',
    model: 'openai/gpt-oss-120b',
    env: 'GROQ_KEY'
  },
  openrouter: {
    url: 'https://openrouter.ai/api/v1/chat/completions',
    model: 'openrouter/free',
    env: 'OPENROUTER_KEY'
  }
};

/* Very small burst limiter. Per-isolate only — pair it with Vercel Firewall or
   a KV-based limiter if the site gets real traffic. */
const hits = new Map();
function rateLimited(ip) {
  const now = Date.now();
  const bucket = hits.get(ip);
  if (!bucket || now - bucket.start > 60_000) {
    hits.set(ip, { start: now, count: 1 });
    return false;
  }
  bucket.count += 1;
  if (hits.size > 5000) hits.clear();
  return bucket.count > 40;
}

function corsHeaders(req) {
  const origin = req.headers.get('origin');
  return {
    'Access-Control-Allow-Origin': origin || '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    Vary: 'Origin'
  };
}

function reply(req, body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', ...corsHeaders(req) }
  });
}

function sameOrigin(req) {
  const origin = req.headers.get('origin');
  if (!origin) return true;
  try {
    const host = req.headers.get('host') || req.headers.get('x-forwarded-host');
    return !!host && new URL(origin).host === host;
  } catch {
    return false;
  }
}

export default async function handler(req) {
  if (req.method === 'OPTIONS') return new Response(null, { status: 204, headers: corsHeaders(req) });
  if (req.method !== 'POST') return reply(req, { error: 'method_not_allowed' }, 405);
  if (!sameOrigin(req)) return reply(req, { error: 'origin_not_allowed' }, 403);

  const ip = (req.headers.get('x-forwarded-for') || '').split(',')[0].trim() || 'unknown';
  if (rateLimited(ip)) return reply(req, { error: 'rate_limited' }, 429);

  const declared = Number(req.headers.get('content-length') || 0);
  if (declared > 3_000_000) return reply(req, { error: 'payload_too_large' }, 413);

  let body;
  try {
    body = await req.json();
  } catch {
    return reply(req, { error: 'bad_json' }, 400);
  }

  const provider = PROVIDERS[body.provider === 'openrouter' ? 'openrouter' : 'groq'];
  const key = process.env[provider.env];
  if (!key) return reply(req, { error: 'missing_server_key', variable: provider.env }, 503);

  const messages = Array.isArray(body.messages) ? body.messages.slice(0, 6) : [];
  if (!messages.length) return reply(req, { error: 'missing_messages' }, 400);
  for (const m of messages) {
    if (typeof m?.content === 'string' && m.content.length > 60_000) {
      return reply(req, { error: 'message_too_large' }, 413);
    }
  }

  const jsonMode = body.jsonMode === true;
  const stream = body.stream === true;
  const requested = Number(body.maxTokens);
  const maxTokens = Number.isFinite(requested)
    ? Math.min(Math.max(Math.round(requested), 64), 1200)
    : (jsonMode ? 600 : 900);

  const payload = {
    model: provider.model,
    messages,
    temperature: 0.1,
    stream
  };
  if (provider === PROVIDERS.groq) {
    payload.reasoning_effort = 'low';
    payload.max_completion_tokens = maxTokens;
    if (jsonMode) payload.response_format = { type: 'json_object' };
  }

  let upstream;
  try {
    upstream = await fetch(provider.url, {
      method: 'POST',
      headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  } catch {
    return reply(req, { error: 'upstream_unreachable' }, 502);
  }

  const headers = stream
    ? { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-store', ...corsHeaders(req) }
    : { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', ...corsHeaders(req) };

  return new Response(upstream.body, { status: upstream.status, headers });
}
