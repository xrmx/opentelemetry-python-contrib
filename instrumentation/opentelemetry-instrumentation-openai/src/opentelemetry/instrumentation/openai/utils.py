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


def extract_content(choice):
    if not hasattr(choice, "message"):
        return ""

    # Check if choice.message exists and has a content attribute
    if getattr(choice.message, "content", None) is not None:
        return choice.message.content

    # Check if choice.message has tool_calls and extract information accordingly
    if getattr(choice.message, "tool_calls", None) is not None:
        result = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in choice.message.tool_calls
        ]
        return result

    # Check if choice.message has a function_call and extract information accordingly
    if getattr(choice.message, "function_call", None) is not None:
        return {
            "name": choice.message.function_call.name,
            "arguments": choice.message.function_call.arguments,
        }

    # Return an empty string if none of the above conditions are met
    return ""
