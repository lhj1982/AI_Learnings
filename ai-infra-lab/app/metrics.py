from prometheus_client import Counter, Histogram


AI_REQUESTS = Counter(
    "ai_requests_total",
    "Total number of AI model requests",
    ["provider", "model", "status"],
)

AI_REQUEST_LATENCY = Histogram(
    "ai_request_duration_seconds",
    "Duration of AI model requests",
    ["provider", "model"],
)

AI_INPUT_TOKENS = Counter(
    "ai_input_tokens_total",
    "Total number of input tokens",
    ["provider", "model"],
)

AI_OUTPUT_TOKENS = Counter(
    "ai_output_tokens_total",
    "Total number of output tokens",
    ["provider", "model"],
)

AI_FALLBACKS = Counter(
    "ai_fallback_total",
    "Number of times one provider falls back to another",
    ["from_provider", "to_provider"],
)