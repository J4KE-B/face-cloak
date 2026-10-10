# face-cloak web

Frontend of the face-cloak Multimedia Forensics project: landing page, live demo, results, about.
Next.js 16 (App Router), React 19, TypeScript (strict), Tailwind CSS 4, Framer Motion, lucide-react. No analytics, no cookies.

## Develop

```bash
cd web
npm install
npm run dev          # http://localhost:3000
npm run lint
npm run build && npm run start
```

## Pages

| Route | What |
|---|---|
| `/` | Landing: hero, threat, 7-step pipeline, headline numbers (only if `public/results/summary.json` exists) |
| `/demo` | Live demo (not indexed, see `app/robots.ts`); the browser talks **directly** to the Python backend (no Next.js API routes) |
| `/results` | Tables and figures from `public/results/summary.json` (placeholder until it exists) |
| `/about` | Method, threat model, limitations, ethics, team, credits |

## Backend connection

`/demo` asks for a backend URL and passcode (`DEMO_KEY`). Accepted URLs only: `https://*.trycloudflare.com`,
`http://localhost[:port]`, `http://127.0.0.1[:port]` (and `mock`). `/demo?backend=<url>` pre-fills the URL (the query
parameter is removed from the address bar; the passcode field is cleared whenever the backend URL is changed to a
different host and a stored passcode is never reused for a link to a new host, so a passcode is only ever sent to the host
it was typed for; hosts that arrived via a link need an extra confirmation before the first upload; the passcode is never
put in a URL). Address and passcode are kept in `localStorage`. `/health` is polled every 10 s, jobs every 1.5 s.

Configuration lives in `lib/config.ts`. Every backend response is validated at runtime in `lib/api.ts` (image fields must be
`data:image/(png|jpeg);base64,...`, target links must be https, numbers/strings are type-checked).

Types in `lib/types.ts` mirror `server/schema.py`; the typed client is `lib/api.ts`.

## Mock mode (no backend needed)

Enter `mock` as the backend URL (any passcode). `lib/mock.ts` implements the whole contract in the browser: placeholder
targets drawn on canvas, simulated queue and progress (about 9 s), and a result built from your uploaded photo (noise
cloak, x10 perturbation, SYNTHETIC-labelled fake swaps, canned scores). Passcode `wrong` returns 401, `busy` returns 503,
`limit` returns 429 with a Retry-After countdown. Numbers are canned and labelled as mock in the verdict note.

## Results data

`uv run python scripts/export_web_results.py` (repo root) writes `web/public/results/summary.json` and the PNG figures.
Figures whose names look like qualitative grids are never displayed.

## Deploy (Vercel)

Import the GitHub repo, set **Root Directory = `web/`**, framework Next.js, defaults otherwise. No environment variables are
needed. Security headers (CSP with `connect-src` limited to trycloudflare/localhost/127.0.0.1, no `unsafe-eval` in
production, COOP same-origin, no workers/frames) are set in `next.config.ts`. The backend must allow the origin `https://face-cloak.vercel.app` (`DEMO_SITE`).
