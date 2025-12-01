from prometheus_client import Counter, Histogram, Gauge

GEN_REQUESTS = Counter("gen_requests_total", "Total generation requests", ["model"])
GEN_FAILURES = Counter("gen_failures_total", "Total generation failures", ["model"])
GEN_LAT = Histogram("gen_latency_seconds", "Generation latency",
                    buckets=(0.5, 1, 2, 4, 8, 16, 32),
                    labelnames=("model",))
QUEUE_DEPTH = Gauge("queue_depth", "Requests in queue")
