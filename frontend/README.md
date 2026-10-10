# Regression interface

Next.js interface for uploading a dataset, configuring a regression model and making predictions. The browser uses same-origin `/api` requests; Next.js forwards them to the FastAPI backend.

## Development

Use Node.js 24 LTS. Start the backend as described in the [running guide](../RUNNING.md), then run:

```bash
npm ci
npm run dev
```

Open <http://localhost:3000>. The backend defaults to `http://127.0.0.1:8000`. Set `BACKEND_URL` in `.env.local` to change it.

## Production build

```bash
npm run build
npm start
```

`BACKEND_URL` is resolved when the production build is created; rebuild after changing it. The Dockerfile builds a standalone server with the backend address `http://backend:8000`. It can be changed through the Docker build argument `BACKEND_URL`.

## Checks

```bash
npm run lint
npm test
npm run build
```
