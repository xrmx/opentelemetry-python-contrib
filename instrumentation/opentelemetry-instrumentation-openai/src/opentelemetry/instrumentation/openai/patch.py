# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Optional

from openai import NOT_GIVEN

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Tracer
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry.trace.status import Status, StatusCode

from .span_attributes import LLMSpanAttributes, SpanAttributes
from .utils import extract_content, silently_fail


def chat_completions_create(original_method, version, tracer: Tracer):
    """Wrap the `create` method of the `ChatCompletion` class to trace it."""

    def traced_method(wrapped, instance, args, kwargs):
        llm_prompts = []
        for item in kwargs.get("messages", []):
            tools = get_tool_calls(item)
            if tools is not None:
                tool_calls = []
                for tool_call in tools:
                    tool_call_dict = {
                        "id": getattr(tool_call, "id", ""),
                        "type": getattr(tool_call, "type", ""),
                    }
                    if hasattr(tool_call, "function"):
                        tool_call_dict["function"] = {
                            "name": getattr(tool_call.function, "name", ""),
                            "arguments": getattr(
                                tool_call.function, "arguments", ""
                            ),
                        }
                    tool_calls.append(tool_call_dict)
                llm_prompts.append(tool_calls)
            else:
                llm_prompts.append(item)

        span_attributes = {
            **get_llm_request_attributes(kwargs, prompts=llm_prompts),
        }

        attributes = LLMSpanAttributes(**span_attributes)

        span_name = f"{attributes.gen_ai_operation_name} {attributes.gen_ai_request_model}"

        span = tracer.start_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            context=set_span_in_context(trace.get_current_span()),
        )
        _set_input_attributes(span, kwargs, attributes)

        try:
            result = wrapped(*args, **kwargs)
            if is_streaming(kwargs):
                return StreamWrapper(
                    result,
                    span,
                    function_call=kwargs.get("functions") is not None,
                    tool_calls=kwargs.get("tools") is not None,
                )
            else:
                _set_response_attributes(span, kwargs, result)
                span.end()
                return result

        except Exception as error:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()
            raise

    return traced_method


def get_tool_calls(item):
    if isinstance(item, dict):
        return item.get("tool_calls")
    else:
        return getattr(item, "tool_calls", None)


def is_given(value):
    return value is not None and value != NOT_GIVEN


@silently_fail
def _set_input_attributes(span, kwargs, attributes: LLMSpanAttributes):
    tools = []

    if is_given(kwargs.get("functions")):
        for function in kwargs.get("functions", []):
            tools.append(
                json.dumps({"type": "function", "function": function})
            )

    if is_given(kwargs.get("tools")):
        tools.append(json.dumps(kwargs.get("tools")))

    if tools:
        set_span_attribute(span, SpanAttributes.LLM_TOOLS, json.dumps(tools))

    # FIXME: double check pydantic usage
    for field, value in attributes.model_dump(by_alias=True).items():
        set_span_attribute(span, field, value)


@silently_fail
def _set_response_attributes(span, kwargs, result):
    set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, result.model)
    if getattr(result, "choices", None) is not None:
        responses = [
            {
                "role": (
                    choice.message.role
                    if choice.message and choice.message.role
                    else "assistant"
                ),
                "content": extract_content(choice),
                **(
                    {
                        "content_filter_results": choice[
                            "content_filter_results"
                        ]
                    }
                    if "content_filter_results" in choice
                    else {}
                ),
            }
            for choice in result.choices
        ]
        set_event_completion(span, responses)

    if is_given(getattr(result, "system_fingerprint", None)):
        set_span_attribute(
            span,
            SpanAttributes.LLM_SYSTEM_FINGERPRINT,
            result.system_fingerprint,
        )
    # Get the usage
    if getattr(result, "usage", None) is not None:
        if result.usage is not None:
            set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                result.usage.prompt_tokens,
            )
            set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                result.usage.completion_tokens,
            )
            set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                result.usage.total_tokens,
            )


def set_event_prompt(span: Span, prompt):
    span.add_event(
        name=SpanAttributes.LLM_CONTENT_PROMPT,
        attributes={
            SpanAttributes.LLM_PROMPTS: prompt,
        },
    )


def set_span_attributes(span: Span, attributes: dict):
    for field, value in attributes.model_dump(by_alias=True).items():
        set_span_attribute(span, field, value)


def set_event_completion(span: Span, result_content):
    span.add_event(
        name=SpanAttributes.LLM_CONTENT_COMPLETION,
        attributes={
            SpanAttributes.LLM_COMPLETIONS: json.dumps(result_content),
        },
    )


def set_span_attribute(span: Span, name, value):
    if value is not None:
        if value != "" or value != NOT_GIVEN:
            if name == SpanAttributes.LLM_PROMPTS:
                set_event_prompt(span, value)
            else:
                span.set_attribute(name, value)
    return


def is_streaming(kwargs):
    stream: Optional[bool] = kwargs.get("stream")
    return stream and stream != NOT_GIVEN


def get_llm_request_attributes(
    kwargs, prompts=None, model=None, operation_name="chat"
):

    user = kwargs.get("user")
    if prompts is None:
        prompts = (
            [{"role": user or "user", "content": kwargs.get("prompt")}]
            if "prompt" in kwargs
            else None
        )
    top_k = (
        kwargs.get("n")
        or kwargs.get("k")
        or kwargs.get("top_k")
        or kwargs.get("top_n")
    )

    top_p = kwargs.get("p") or kwargs.get("top_p")
    tools = kwargs.get("tools")
    return {
        SpanAttributes.LLM_OPERATION_NAME: operation_name,
        SpanAttributes.LLM_REQUEST_MODEL: model or kwargs.get("model"),
        SpanAttributes.LLM_IS_STREAMING: kwargs.get("stream"),
        SpanAttributes.LLM_REQUEST_TEMPERATURE: kwargs.get("temperature"),
        SpanAttributes.LLM_TOP_K: top_k,
        SpanAttributes.LLM_PROMPTS: json.dumps(prompts) if prompts else None,
        SpanAttributes.LLM_USER: user,
        SpanAttributes.LLM_REQUEST_TOP_P: top_p,
        SpanAttributes.LLM_REQUEST_MAX_TOKENS: kwargs.get("max_tokens"),
        SpanAttributes.LLM_SYSTEM_FINGERPRINT: kwargs.get(
            "system_fingerprint"
        ),
        SpanAttributes.LLM_PRESENCE_PENALTY: kwargs.get("presence_penalty"),
        SpanAttributes.LLM_FREQUENCY_PENALTY: kwargs.get("frequency_penalty"),
        SpanAttributes.LLM_REQUEST_SEED: kwargs.get("seed"),
        SpanAttributes.LLM_TOOLS: json.dumps(tools) if tools else None,
        SpanAttributes.LLM_TOOL_CHOICE: kwargs.get("tool_choice"),
        SpanAttributes.LLM_REQUEST_LOGPROPS: kwargs.get("logprobs"),
        SpanAttributes.LLM_REQUEST_LOGITBIAS: kwargs.get("logit_bias"),
        SpanAttributes.LLM_REQUEST_TOP_LOGPROPS: kwargs.get("top_logprobs"),
    }


class StreamWrapper:
    span: Span

    def __init__(
        self,
        stream,
        span,
        prompt_tokens=None,
        function_call=False,
        tool_calls=False,
    ):
        self.stream = stream
        self.span = span
        self.prompt_tokens = prompt_tokens
        self.function_call = function_call
        self.tool_calls = tool_calls
        self.result_content = []
        self.completion_tokens = 0
        self._span_started = False
        self.setup()

    def setup(self):
        if not self._span_started:
            self._span_started = True

    def cleanup(self):
        if self._span_started:
            set_span_attribute(
                self.span,
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                self.prompt_tokens,
            )
            set_span_attribute(
                self.span,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                self.completion_tokens,
            )
            set_span_attribute(
                self.span,
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                self.prompt_tokens + self.completion_tokens,
            )
            set_event_completion(
                self.span,
                [
                    {
                        "role": "assistant",
                        "content": "".join(self.result_content),
                    }
                ],
            )

            self.span.set_status(StatusCode.OK)
            self.span.end()
            self._span_started = False

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self.stream)
            self.process_chunk(chunk)
            return chunk
        except StopIteration:
            self.cleanup()
            raise

    def process_chunk(self, chunk):
        if hasattr(chunk, "model") and chunk.model is not None:
            set_span_attribute(
                self.span,
                SpanAttributes.LLM_RESPONSE_MODEL,
                chunk.model,
            )

        if hasattr(chunk, "choices") and chunk.choices is not None:
            content = []
            if not self.function_call and not self.tool_calls:
                for choice in chunk.choices:
                    if choice.delta and choice.delta.content is not None:
                        content = [choice.delta.content]
            elif self.function_call:
                for choice in chunk.choices:
                    if (
                        choice.delta
                        and choice.delta.function_call is not None
                        and choice.delta.function_call.arguments is not None
                    ):
                        content = [choice.delta.function_call.arguments]
            elif self.tool_calls:
                for choice in chunk.choices:
                    if choice.delta and choice.delta.tool_calls is not None:
                        toolcalls = choice.delta.tool_calls
                        content = []
                        for tool_call in toolcalls:
                            if (
                                tool_call
                                and tool_call.function is not None
                                and tool_call.function.arguments is not None
                            ):
                                content.append(tool_call.function.arguments)

            if content:
                self.result_content.append(content[0])

        if hasattr(chunk, "text"):
            content = [chunk.text]

            if content:
                self.result_content.append(content[0])

        if getattr(chunk, "usage"):
            self.completion_tokens = chunk.usage.completion_tokens
            self.prompt_tokens = chunk.usage.prompt_tokens
