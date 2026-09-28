
> [!WARNING]
> GenAI instrumentations have been moved to the
> [opentelemetry-python-genai](https://github.com/open-telemetry/opentelemetry-python-genai)
> repository, which is now their home for new development and releases. The copies that
> still live in this directory only receive security patches and will be removed from
> here in the future.

| Instrumentation | Supported Packages | Metrics support | Semconv status |
| --------------- | ------------------ | --------------- | -------------- |
| [opentelemetry-instrumentation-genai-openai-agents](https://github.com/open-telemetry/opentelemetry-python-genai/tree/main/instrumentation/opentelemetry-instrumentation-genai-openai-agents) | openai-agents >= 0.3.3 | No | development
| [opentelemetry-instrumentation-genai-openai](https://github.com/open-telemetry/opentelemetry-python-genai/tree/main/instrumentation/opentelemetry-instrumentation-genai-openai) | openai >= 1.26.0 | Yes | development
| [opentelemetry-instrumentation-vertexai](./opentelemetry-instrumentation-vertexai) | google-cloud-aiplatform >= 1.64 | No | development
