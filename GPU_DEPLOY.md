# Running on a GPU server

This repo ships the **architecture only** — code, the trained ML classifier
artifacts (`ml/models/*.joblib`), and the knowledge base
(`data/reason_code_kb.json`). It does **not** ship the raw training rows
(`data/train.jsonl` / `val.jsonl` / `test.jsonl` are git-ignored) or any Ollama
model weights — those live outside git and are handled separately, as below.

Nothing in this stack requires the GPU directly. The GPU is used by **Ollama**,
which this app talks to over plain HTTP (`OLLAMA_HOST`) — the FastAPI service,
the ML classifier, and the KB all run on CPU. So "deploying to the GPU server"
really means: get Ollama running there with GPU acceleration, get this app
running alongside it, and point one at the other.

---

## 1. Prerequisites on the GPU server

- Linux with an NVIDIA GPU + driver already installed (`nvidia-smi` should work)
- Python 3.10+
- git

## 2. Clone this repo

```bash
git clone https://github.com/Sivapriya785-Zensar/SLM--GPU-export.git
cd SLM--GPU-export
```

## 3. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The ML classifier artifacts (`ml/models/*.joblib`) are already in the repo —
**no retraining needed**. (If you ever do want to retrain: you'd need to copy
`data/train.jsonl` etc. over separately, since those aren't in git, then run
`python ml/train.py`.)

## 4. Install and start Ollama with GPU support

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &          # or run it as a systemd service, see step 7
```

Ollama auto-detects the NVIDIA GPU via CUDA — no code changes needed here.
Confirm it's actually using the GPU once a model is loaded:

```bash
nvidia-smi     # should show an `ollama` process using GPU memory
```

## 5. Get your fine-tuned model(s) onto this machine

The model weights are **not** in this repo. Bring them over one of two ways:

**A. Push the GGUF + Modelfile from your dev machine and register locally:**

```bash
# on the GPU server, after copying the .gguf + Modelfile over (scp/rsync):
ollama create dispute-phi3-4ep -f Modelfile
```

**B. If the model is already pushed to an Ollama-compatible registry**, just:

```bash
ollama pull <your-namespace>/dispute-phi3-4ep
```

Repeat for every model you want selectable from the frontend's Settings panel
(e.g. `dispute-phi3-4ep`, `dispute-phi3-curated`, or whatever you train next).
Verify they're all visible:

```bash
ollama list
```

## 6. Configure and run the app

Set env vars as needed (all optional, sensible defaults — see `slm/client.py`
/ `app/main.py`):

```bash
export SLM_MODEL="dispute-phi3-4ep:latest"   # default active model
export OLLAMA_HOST="http://localhost:11434"  # only needed if Ollama isn't local

# The Settings picker's model list is a static, comma-separated env var — NOT a live
# query against Ollama. (An earlier version called Ollama's /api/tags on every page
# load; on Windows that call could hang indefinitely if Ollama's HTTP listener ever
# wedged, taking the whole picker down with it.) Set this to whatever `ollama list`
# actually shows as installed on THIS machine:
export SLM_AVAILABLE_MODELS="dispute-slm:latest,disputeslm-v2:latest"
```

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8010
```

Open `http://<gpu-server-ip>:8010`. The frontend is served by the same
FastAPI process — no separate frontend server or build step.

**Switching models without a restart:** open the "Model Settings" panel in
the sidebar — it lists whatever `SLM_AVAILABLE_MODELS` names (via
`GET /api/models`) and lets you pick the active one (`POST /api/settings`),
live, without restarting the server. Update `SLM_AVAILABLE_MODELS` (and
restart the app) whenever you pull/create a new model on this machine.

## 7. Running it as a persistent service (recommended over `&`)

Two systemd units, so both survive a reboot / SSH disconnect:

```ini
# /etc/systemd/system/ollama.service
[Unit]
Description=Ollama
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
Restart=always
User=%i

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/dispute-slm.service
[Unit]
Description=Dispute SLM API
After=ollama.service
Requires=ollama.service

[Service]
WorkingDirectory=/path/to/SLM--GPU-export
Environment=SLM_MODEL=dispute-phi3-4ep:latest
ExecStart=/path/to/SLM--GPU-export/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8010
Restart=always
User=%i

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ollama dispute-slm
```

## 8. Exposing it

Port 8010 is plain HTTP with no auth built in — put it behind a reverse proxy
(nginx/Caddy) with TLS and, if this will be reachable outside a trusted
network, some access control, before exposing it beyond localhost/VPN.

## 9. Verify it's actually working

```bash
curl http://localhost:8010/api/health
# {"status":"ok", "slm_available": true, "slm_model": "dispute-phi3-4ep:latest", ...}

curl http://localhost:8010/api/models
# {"active": "dispute-phi3-4ep:latest", "models": [...]}
```

If `slm_available` is `false`, Ollama isn't reachable or the model named in
`SLM_MODEL` isn't pulled/created yet — check `ollama list` and `OLLAMA_HOST`.
