"""The aggregator: one model reads both answers and writes one combined answer.

    python -m moa ask --peer <B> --aggregate "..."

The aggregator is a third call to a model (by default this machine's model),
without tools. It gets the question and both answers. In agent mode it also
gets each answer's tool record, written by the code and not by the models, so
it can tell which facts came from a tool that really ran and which did not.

It is a hint, not a check: the aggregator is a model too and can still make
mistakes. Both original answers are always shown next to the combined one.
"""

from __future__ import annotations

import time
from typing import List, Optional

from .core import Answer
from .ollama_client import OllamaClient, OllamaError
from .output import describe_tools

INSTRUCTIONS = """\
You get one question and two answers to it, written by two different AI models.
Write one combined answer to the question.

Rules:
- Use only what is in the two answers. Do not add new facts.
- Where the answers agree, say it once.
- Where they disagree, say so plainly, and say which answer is better supported and why.
- Each answer comes with a tool record written by the program, not by the model.
  A fact about files, emails or other resources is supported only if that answer's
  tool record shows a successful tool call that could have returned it.
  Mark any such fact without that support as "(unverified)".
- End with one line: "Agreement: full", "Agreement: partial" or "Agreement: none".
"""


def build_prompt(question: str, answers: List[Answer]) -> str:
    """The aggregator's prompt: instructions, the question, then each answer with its record."""
    parts = [INSTRUCTIONS, f"QUESTION:\n{question}"]
    for number, answer in enumerate(answers, start=1):
        parts.append(f"ANSWER {number} (machine {answer.node}, model {answer.model})\n"
                     f"Tool record: {_record(answer)}\n"
                     f"Text:\n{(answer.text or '').strip()}")
    return "\n\n".join(parts)


def _record(answer: Answer) -> str:
    if answer.tools is None:
        return "no tools were available (plain model answer)"
    record = describe_tools(answer.tools)
    if answer.warnings:
        record += "; servers that could not be used: " + "; ".join(answer.warnings)
    return record


def aggregate(question: str, answers: List[Answer], ollama: OllamaClient,
              model: Optional[str] = None, node_name: str = "local") -> Optional[Answer]:
    """Combine the answers with one more model call on this machine.

    Returns None when there is nothing to combine (fewer than two good
    answers); the caller shows why. A failed aggregator call is returned as a
    failed Answer, like any other answer.
    """
    good = [a for a in answers if a.ok]
    if len(good) < 2:
        return None
    started = time.perf_counter()
    try:
        result = ollama.generate(build_prompt(question, good), model)
    except OllamaError as exc:
        return Answer("aggregate", node_name, False, error=str(exc),
                      elapsed_s=time.perf_counter() - started)
    return Answer("aggregate", node_name, True, result.model, result.text, elapsed_s=result.elapsed_s,
                  timing=result.stats)
