#!/usr/bin/env python3
"""
Simple interactive chat using LiteLLM (streaming).
Requires:
    pip install litellm python-dotenv
Place your environment variables in a .env file, for example:
    OPENAI_API_KEY="sk-..."
    OPENAI_API_BASE="https://api.openai.com/v1"   # or your OpenAI-compatible base url
You can also set LITELLM_MODEL in the env, otherwise defaults to openai/gpt-3.5-turbo.
"""

import os
import sys
from dotenv import load_dotenv

# load .env
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in environment. Make sure .env is configured.", file=sys.stderr)

MODEL = os.getenv("LITELLM_MODEL", "openai/gemma-3")

from litellm import completion  

def extract_chunk_text(part):
    """
    Try a few common ways to get token text from a streaming chunk.
    This handles variations in the chunk structure across providers.
    """
    try:
        # Many examples return objects where .choices[0].delta.content exists
        c = part.choices[0].delta
        # try attribute first
        if hasattr(c, "content"):
            return c.content or ""
        # then dict-style
        if isinstance(c, dict):
            return c.get("content", "") or ""
    except Exception:
        pass

    # Some providers may include finished message fragments
    try:
        msg = part.choices[0].message
        if hasattr(msg, "content"):
            return msg.content or ""
        if isinstance(msg, dict):
            return msg.get("content", "") or ""
    except Exception:
        pass

    # fallback to safest str conversion
    try:
        return str(part)
    except Exception:
        return ""

def main():
    print(f"LiteLLM interactive chat (model={MODEL}). Type 'quit' or 'exit' to stop.\n")

    # conversation history as OpenAI-style messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        # add user turn to history
        messages.append({"role": "user", "content": user_input})

        # call litellm with streaming=True to print iteratively as tokens arrive
        try:
            print("Assistant: ", end="", flush=True)
            chunks_text = []
            # stream=True returns an iterator of chunks
            response_iter = completion(model=MODEL, messages=messages, stream=True)

            for part in response_iter:
                token = extract_chunk_text(part)
                if token:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    chunks_text.append(token)

            # final newline after the streamed response
            print()

            # record assistant reply in messages history
            assistant_text = "".join(chunks_text).strip()
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
            else:
                # fallback: try to call without streaming to get a stable response
                resp = completion(model=MODEL, messages=messages)
                text = ""
                try:
                    text = resp.choices[0].message.content
                except Exception:
                    text = str(resp)
                print("Assistant (fallback):")
                print(text)
                messages.append({"role": "assistant", "content": text})

        except Exception as e:
            print("\n[Error] Request failed:", e, file=sys.stderr)
            # optionally remove the last user turn to avoid poisoning history, or keep it
            # messages.pop()  # uncomment to remove last user message from history
            continue


if __name__ == "__main__":
    main()
