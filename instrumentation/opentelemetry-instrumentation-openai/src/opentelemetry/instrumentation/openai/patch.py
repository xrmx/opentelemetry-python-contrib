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
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as genai,
)
from opentelemetry.trace import Span, SpanKind, Tracer
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry.trace.status import Status, StatusCode

from .utils import extract_content


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

        attributes = {
            **get_llm_request_attributes(kwargs),
        }

        span_name = f"{attributes.gen_ai_operation_name} {attributes.gen_ai_request_model}"

        span = tracer.start_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            context=set_span_in_context(trace.get_current_span()),
        )

        _set_input_attributes(span, attributes)
        prompts = get_llm_request_prompts(kwargs, llm_prompts)
        if prompts is not None:
            set_event_prompt(span, prompts)

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


def _set_input_attributes(span, attributes):
    for field, value in attributes.items():
        set_span_attribute(span, field, value)


def _set_response_attributes(span, kwargs, result):
    set_span_attribute(span, genai.GEN_AI_RESPONSE_MODEL, result.model)
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
        # FIXME: WIP semconv
        set_span_attribute(
            span,
            "gen_ai.system_fingerprint",
            result.system_fingerprint,
        )

    # Get the usage
    if getattr(result, "usage", None) is not None:
        if result.usage is not None:
            set_span_attribute(
                span,
                genai.GEN_AI_USAGE_PROMPT_TOKENS,
                result.usage.prompt_tokens,
            )
            set_span_attribute(
                span,
                genai.GEN_AI_USAGE_COMPLETION_TOKENS,
                result.usage.completion_tokens,
            )


def set_event_prompt(span: Span, prompt):
    span.add_event(
        name="gen_ai.content.prompt",
        attributes={
            genai.GEN_AI_PROMPT: prompt,
        },
    )


def set_event_completion(span: Span, result_content):
    span.add_event(
        name="gen_ai.content.completion",
        attributes={
            genai.GEN_AI_COMPLETION: json.dumps(result_content),
        },
    )


def set_span_attribute(span: Span, name, value):
    if value is not None:
        if value != "" or value != NOT_GIVEN:
            span.set_attribute(name, value)


def is_streaming(kwargs):
    stream: Optional[bool] = kwargs.get("stream")
    return stream and stream != NOT_GIVEN


def get_llm_request_prompts(kwargs, prompts):
    user = kwargs.get("user")
    if prompts is None:
        prompts = (
            [{"role": user or "user", "content": kwargs.get("prompt")}]
            if "prompt" in kwargs
            else None
        )
    return json.dumps(prompts) if prompts else None


def get_llm_request_attributes(kwargs, model=None, operation_name="chat"):

    top_k = (
        kwargs.get("n")
        or kwargs.get("k")
        or kwargs.get("top_k")
        or kwargs.get("top_n")
    )

    top_p = kwargs.get("p") or kwargs.get("top_p")
    return {
        genai.GEN_AI_OPERATION_NAME: operation_name,
        genai.GEN_AI_REQUEST_MODEL: model or kwargs.get("model"),
        genai.GEN_AI_REQUEST_TEMPERATURE: kwargs.get("temperature"),
        genai.GEN_AI_REQUEST_TOP_P: top_p,
        genai.GEN_AI_REQUEST_MAX_TOKENS: kwargs.get("max_tokens"),
        genai.GEN_AI_PRESENCE_PENALTY: kwargs.get("presence_penalty"),
        genai.GEN_AI_FREQUENCY_PENALTY: kwargs.get("frequency_penalty"),
        genai.GEN_AI_TOP_K: top_k,
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
                genai.GEN_AI_USAGE_INPUT_TOKENS,
                self.prompt_tokens,
            )
            set_span_attribute(
                self.span,
                genai.GEN_AI_USAGE_OUTPUT_TOKENS,
                self.completion_tokens,
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
                genai.GEN_AI_RESPONSE_MODEL,
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
