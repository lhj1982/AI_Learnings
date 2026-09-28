# Install vllm

## install vLLM-Metal on macpro
```
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh \
  | bash -s -- --stable

source ~/.venv-vllm-metal/bin/activate

python --version
which python
vllm --version

# start qwen model
vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8001

or

vllm serve Qwen/Qwen2.5-1.5B-Instruct

or if want to add api authentication

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8001 \
    --api-key my-local-secret
```

Verify
curl http://localhost:8001/v1/models

if have api key
add to .env
```
LLM_API_KEY=my-local-secret
```


## Test

```
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Explain Kubernetes in one sentence"}'
```


## Add the metrics AI Infra engineers care about
For every request record:

```
model
provider
request latency
TTFT
input tokens
output tokens
tokens/sec
status
error
cost
```
For example:
```
{
  "provider": "deepseek",
  "model": "deepseek-v4-pro",
  "latency_ms": 2431,
  "input_tokens": 534,
  "output_tokens": 197,
  "cost_usd": 0.0004
}
```
For local vLLM also record:
```
GPU utilization
VRAM usage
running requests
waiting requests
KV cache utilization
generation throughput
```
This is very important.

With external API:
```
Optimization target:

       $
 cost / request
```
With self-hosting:
```
Optimization targets:

GPU utilization
tokens/sec
requests/sec
latency
cost/GPU-hour
```
Then you can compare:
```
              DeepSeek API       Qwen/vLLM

Cost          token based        GPU-hour
Control       lower              high
Operations    easy               your responsibility
Scaling       provider handles   you handle
Model         provider model     your model
Latency       network/API        potentially local
Privacy       external           self-hosted
```


## Dockerize

```
docker build -t ai-infra-lab .

docker run \
  --name ai-infra-api \
  --env-file .env \
  -p 8000:8000 \
  ai-infra-lab

# start vLLM Metal
vllm serve Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name qwen-local
```

Or using docker composer
```
docker compose up --build
```


# Design

                         Mac
┌────────────────────────────────────────────────────┐
│                                                    │
│   Docker Compose                                   │
│                                                    │
│   ┌──────────┐       scrape       ┌────────────┐   │
│   │ FastAPI  │◄───────────────────│ Prometheus │   │
│   │ :8000    │   /metrics         │ :9090      │   │
│   └────┬─────┘                    └──────┬─────┘   │
│        │                                │ query    │
│        │                                ▼          │
│        │                         ┌────────────┐     │
│        │                         │ Grafana    │     │
│        │                         │ :3000      │     │
│        │                         └────────────┘     │
│        │                                           │
│        │ host.docker.internal                      │
│        ▼                                           │
│   vLLM/Qwen :8001  ← native on Mac                 │
│                                                    │
└────────────────────────────────────────────────────┘