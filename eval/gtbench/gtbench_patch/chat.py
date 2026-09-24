"""Chat backend for GTBench (gamingbench.chat.chat), rewritten to talk to local
vLLM OpenAI-compatible servers via the modern OpenAI python client. Replaces the
original langchain implementation (which is the ONLY place gamingbench imports
langchain, so this fork needs neither langchain nor the pinned openai==1.3.5).

Model routing
-------------
Candidate model configs use a path of the form:

    local/<served_name>[:think|:nothink]

  * <served_name>  -> a key in the LOCAL_LLM_ENDPOINTS env var (JSON), e.g.
        LOCAL_LLM_ENDPOINTS='{"q8b": "http://127.0.0.1:8003/v1"}'
  * :think / :nothink  -> toggles Qwen3's native chain-of-thought via
        extra_body={"chat_template_kwargs": {"enable_thinking": <bool>}}.
    Default (no suffix) is nothink.

In thinking mode the returned text is stripped of everything up to and including
the closing </think> tag, so GTBench's action regex can never latch onto a
bracketed move that appeared inside the reasoning trace.
"""
import json
import os
import re

from openai import OpenAI

_CLIENTS = {}
_THINK_RE = re.compile(r"^.*?</think>", re.DOTALL)


def _get_client(served_name):
    if served_name in _CLIENTS:
        return _CLIENTS[served_name]
    endpoints = json.loads(os.environ.get("LOCAL_LLM_ENDPOINTS", "{}"))
    if served_name not in endpoints:
        raise ValueError(
            f"No endpoint for model '{served_name}'. Set LOCAL_LLM_ENDPOINTS env var. "
            f"Known: {list(endpoints)}")
    client = OpenAI(base_url=endpoints[served_name], api_key="EMPTY", timeout=120)
    _CLIENTS[served_name] = client
    return client


def _parse_model(model):
    """local/<served>[:mode] -> (served_name, enable_thinking: bool)."""
    assert model.startswith("local/"), (
        f"Only local vLLM models are supported in this fork, got '{model}'")
    spec = model[len("local/"):]
    enable_thinking = False
    if ":" in spec:
        spec, mode = spec.rsplit(":", 1)
        enable_thinking = mode.strip().lower() == "think"
    return spec, enable_thinking


def _strip_think(text, enable_thinking):
    if enable_thinking and "</think>" in text:
        text = _THINK_RE.sub("", text, count=1)
    return text.strip()


def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)


def chat_llm(messages, model, temperature, max_tokens, n, timeout, stop,
             return_tokens=False, chat_seed=0):
    served_name, enable_thinking = _parse_model(model)
    client = _get_client(served_name)
    extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

    responses = []
    raw_responses = []
    total_completion_tokens = 0
    total_prompt_tokens = 0
    for _ in range(max(n, 1)):
        resp = client.chat.completions.create(
            model=served_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=[stop] if stop is not None else None,
            timeout=timeout,
            extra_body=extra_body,
        )
        content = resp.choices[0].message.content or ""
        raw_responses.append(content)                       # pre-strip: keeps the <think> reasoning
        responses.append(_strip_think(content, enable_thinking))
        if resp.usage is not None:
            total_completion_tokens += resp.usage.completion_tokens
            total_prompt_tokens += resp.usage.prompt_tokens

    return {
        'generations': responses,
        'raw_generations': raw_responses,   # for the gold-free process verifier; parsing still uses `generations`
        'completion_tokens': total_completion_tokens,
        'prompt_tokens': total_prompt_tokens,
    }
