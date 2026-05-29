# RunPod Deployment Guide

This deploys only the API stack to RunPod. The Tkinter UI stays local and calls the RunPod HTTPS proxy URL.

## Architecture

```text
Local Tkinter UI
  -> https://<pod-id>-8000.proxy.runpod.net
  -> RunPod GPU container
      -> FastAPI Seamless M4T API on port 8000
      -> local Postgres + pgvector on 127.0.0.1:5432
      -> persistent DB data at /workspace/polyglot-postgres
```

The container uses HTTP internally. RunPod exposes the public endpoint over HTTPS.

## 1. Build And Push The Image

Replace `<your-registry-user>` with your Docker Hub or GHCR namespace.

```powershell
cd C:\Users\afrca\Desktop\School\Licenta\polyglot-api

docker build -f dockerfile.runpod -t <your-registry-user>/polyglot-api:runpod .
docker push <your-registry-user>/polyglot-api:runpod
```

The image is large because it includes the local Seamless M4T model.

## 2. Create The RunPod Pod

Use **Pods**, not Serverless.

Recommended settings:

```text
Image: <your-registry-user>/polyglot-api:runpod
GPU: RTX 4090 24 GB preferred
Fallback GPU: L4 / A5000 / RTX 3090
Expose HTTP Port: 8000
Persistent Volume Mount: /workspace
Container Disk: 50 GB minimum
Volume Size: 20 GB minimum
```

Environment variables:

```text
WORKER_COUNT=1
GUNICORN_TIMEOUT=600
POLYGLOT_API_TOKEN=<choose-a-demo-secret>
POLYGLOT_SEMANTIC_CACHE_ENABLED=true
POLYGLOT_TRANSCRIPT_MEMORY_ENABLED=true
POLYGLOT_CACHE_STRATEGY=context
POLYGLOT_SIMILARITY_THRESHOLD=0.98
POLYGLOT_TEXT_SIMILARITY_THRESHOLD=0.92
```

`POLYGLOT_API_TOKEN` is optional but recommended for a public demo endpoint.

## 3. Verify Health

Replace `<pod-id>` and token:

```powershell
curl.exe `
  -H "X-Polyglot-Api-Key: <choose-a-demo-secret>" `
  https://<pod-id>-8000.proxy.runpod.net/health
```

Expected values:

```text
status: true
model_loaded: true
processor_loaded: true
device: cuda
semantic_memory.enabled: true
semantic_memory.status: true
semantic_memory.mode: audio+transcript
```

## 4. Configure The Local UI

Edit:

```text
C:\Users\afrca\Desktop\School\Licenta\polyglot-tkinter-app\config.json
```

Set the active API profile to `runpod` and replace the placeholder URL/token:

```json
{
  "api": {
    "profile": "runpod",
    "profiles": {
      "runpod": {
        "base_url": "https://<pod-id>-8000.proxy.runpod.net/",
        "verify_ssl": true,
        "api_key": "<choose-a-demo-secret>"
      }
    }
  }
}
```

Then start the UI locally:

```powershell
cd C:\Users\afrca\Desktop\School\Licenta\polyglot-tkinter-app
C:\Users\afrca\anaconda3\python.exe main.py
```

## 5. Run Benchmark Against RunPod

```powershell
cd C:\Users\afrca\Desktop\School\Licenta\polyglot-api

C:\Users\afrca\anaconda3\python.exe scripts\semantic_benchmark.py `
  --manifest evaluation\demo\manifest.csv `
  --base-url https://<pod-id>-8000.proxy.runpod.net `
  --api-key <choose-a-demo-secret> `
  --strategy context `
  --session-id runpod-demo-001 `
  --output-dir evaluation\demo\results-runpod `
  --use-transcript-memory true
```

## 6. Stop Costs

After the demo or benchmark:

```text
Download/copy evaluation results if needed.
Stop the pod.
Delete the pod and persistent volume if you no longer need the DB contents.
```

Keeping the volume preserves semantic memory but may continue to cost money.

## Operational Notes

- Keep `WORKER_COUNT=1`; extra workers duplicate the large model in GPU memory.
- RunPod DB memory is demo/benchmark memory, not managed production storage.
- Postgres data persists only under `/workspace/polyglot-postgres`.
- If `/health` reports `device: cpu`, the pod was not started with a GPU-enabled runtime.
