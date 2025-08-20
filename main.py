"""
main.py

A compact, production-friendly starting point for an interactive agent + router
that decides whether a user's message is a "specific question" that should be
sent through a RAG flow (search -> validate -> summarize -> answer), or handled
as normal conversational chat.

This file intentionally uses pure-python stubs for external integrations (search,
LLM, vector DB) and documents clear integration points so you can drop in your
preferred providers (OpenAI, Anthropic, Google Custom Search, Bing API, Pinecone,
etc.).

How it works (high level):
- ConversationAgent holds short-term history and handles regular chat replies.
- Router decides whether a user message should be handled by the RAG flow.
- RAGFlow orchestrates the four steps (search, validate, summarize, answer).

Run: python main.py

Replace the TODO stubs with your actual search/summarization/LLM calls.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# --------------------------- Types & Utilities ------------------------------

SearchResult = Dict[str, Any]  # {'title': str, 'url': str, 'snippet': str, 'score': float}


def now_ts() -> float:
    return time.time()


# --------------------------- Router ----------------------------------------

class Router:
    """Decides whether a user message is a 'specific question' (send to RAG flow)
    or a normal conversational message.

    This implementation uses heuristics:
      - presence of question words (who/what/when/where/how/why)
      - presence of words that imply factual or source-backed answers (cite, source,
        reference, stats, evidence, verify)
      - explicit trigger tokens (search:, lookup:, cite:)
      - length and specificity signals (mentions of dates, numbers, names)

    You should tune the weights or replace this with an LLM-based classifier
    (call a small LLM classification prompt) if you need higher accuracy.
    """

    QUESTION_WORDS = re.compile(r"\b(who|what|when|where|why|how|which)\b", re.I)
    TRIGGER_TOKENS = re.compile(r"\b(search:|lookup:|cite:|sources:|verify:)\b", re.I)
    SOURCE_WORDS = re.compile(r"\b(source|cite|reference|evidence|stats|data)\b", re.I)

    def is_specific_question(self, message: str) -> Tuple[bool, Dict[str, float]]:
        """Return (should_route, diagnostic_scores)"""
        msg = message.strip()
        scores: Dict[str, float] = {"question_word": 0.0, "trigger": 0.0, "source_word": 0.0, "length": 0.0}

        if self.QUESTION_WORDS.search(msg):
            scores["question_word"] = 1.0

        if self.TRIGGER_TOKENS.search(msg):
            scores["trigger"] = 2.0

        if self.SOURCE_WORDS.search(msg):
            scores["source_word"] = 0.8

        # if user mentions an exact date, specific numeric ask or asks "exact"/"precise"
        if re.search(r"\b(\d{4}|\d+%|%|\bexactly\b|\bprecisely\b)\b", msg):
            scores["length"] = 0.7

        # Simple aggregation rule — tuned by you
        total = scores["question_word"] + scores["trigger"] + scores["source_word"] + scores["length"]
        should_route = total >= 1.5
        return should_route, scores


# --------------------------- RAG Flow --------------------------------------

class RAGFlow:
    """A simple RAG flow with clear integration points.

    Steps implemented as methods so you can replace them with real integrations.
    Each method returns serializable data so you can independently test components.
    """

    def __init__(self, max_results: int = 8):
        self.max_results = max_results

    # ---- Step 1: Search -------------------------------------------------
    def search(self, question: str) -> List[SearchResult]:
        """
        Replace this with a real search or retrieval implementation:
          - web search (Bing, Google CSE, SerpAPI)
          - vector search in your knowledge base (Pinecone, Milvus, Qdrant)

        Expected return: list of results with at least title, url, snippet, score.
        """
        # ----- STUB: return plausible fake results for demonstration -----
        now = now_ts()
        fake_results = [
            {
                "title": f"Result {i} for: {question}",
                "url": f"https://example.com/result-{i}",
                "snippet": f"Snippet {i} summarizing content relevant to '{question}'...",
                "score": float(1.0 / (i + 1)),
                "fetched_at": now,
            }
            for i in range(self.max_results)
        ]
        return fake_results

    # ---- Step 2: Validate ------------------------------------------------
    def validate(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Apply filtering heuristics on results (remove low quality / duplicate / stale).
        In production you might:
          - check domain reputation
          - check canonical URLs / duplicates
          - call a small model to label truthfulness/relevance
        """
        filtered: List[SearchResult] = []
        seen_domains = set()
        for r in results:
            domain = re.sub(r"https?://(www\.)?", "", r["url"]).split("/")[0]
            if domain in seen_domains:
                continue
            if len(r.get("snippet", "")) < 20:
                continue
            seen_domains.add(domain)
            filtered.append(r)

        return filtered

    # ---- Step 3: Summarize -----------------------------------------------
    def summarize(self, validated: List[SearchResult]) -> List[Dict[str, str]]:
        """
        Condense each validated result to a short, machine-readable summary.
        You may call an LLM here for extraction/summarization or use an extractive
        summarizer.

        Return value: list of {source, summary}
        """
        outputs = []
        for r in validated:
            # STUB: very small, robust summary derived from snippet.
            s = r.get("snippet", "")
            brief = s[:200].rstrip() + ("..." if len(s) > 200 else "")
            outputs.append({"source": r["url"], "title": r["title"], "summary": brief})
        return outputs

    # ---- Step 4: Answer --------------------------------------------------
    def answer(self, question: str, summaries: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Build a final answer from the condensed summaries. This is where you call
        your main LLM to compose a natural-language answer and attach citations.

        Return a dict: {answer: str, citations: List[str], meta: ...}
        """
        # STUB: simple concatenation + light logic
        if not summaries:
            return {"answer": "I couldn't find reliable sources for that.", "citations": []}

        # Compose a short answer using the top 2 summaries
        top = summaries[:2]
        composed = "\n\n".join(f"From {t['title']}: {t['summary']}" for t in top)
        answer_text = f"Short answer to: {question}\n\n{composed}\n\n(See sources below.)"
        citations = [t["source"] for t in summaries]
        return {"answer": answer_text, "citations": citations, "used_count": len(summaries)}

    # ---- Helper: Full run -----------------------------------------------
    def run(self, question: str) -> Dict[str, Any]:
        results = self.search(question)
        validated = self.validate(results)
        summaries = self.summarize(validated)
        final = self.answer(question, summaries)
        # Attach intermediate artifacts for transparency / debugging
        final["artifacts"] = {"results": results, "validated": validated, "summaries": summaries}
        return final


# --------------------------- Conversation Agent ---------------------------

@dataclass
class ConversationAgent:
    """A simple chat agent that keeps the last-N messages and routes questions
    to the RAG flow when appropriate."""

    name: str = "assistant"
    history: List[Dict[str, str]] = field(default_factory=list)
    max_history_tokens: int = 2048  # placeholder for when integrating LLM context limits
    rag_flow: RAGFlow = field(default_factory=RAGFlow)
    router: Router = field(default_factory=Router)

    def append_human(self, text: str) -> None:
        self.history.append({"role": "user", "text": text, "ts": now_ts()})

    def append_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "text": text, "ts": now_ts()})

    def context_summary(self) -> str:
        # Very small context summarizer for prompt context: last 6 messages
        tail = self.history[-6:]
        return "\n".join([f"{m['role']}: {m['text']}" for m in tail])

    def get_reply(self, message: str) -> Dict[str, Any]:
        """Process an incoming user message and either handle it as chat or
        route to RAGFlow (and return the flow output). Returns a dict with keys
        {type, payload, diagnostic} where type is 'chat'|'rag'."""
        self.append_human(message)

        should_route, scores = self.router.is_specific_question(message)

        if should_route:
            # call the RAG flow synchronously
            rag_result = self.rag_flow.run(message)
            final_answer = rag_result["answer"] if isinstance(rag_result, dict) else str(rag_result)
            # attach to history as assistant reply (shortened)
            self.append_assistant(final_answer)
            return {"type": "rag", "payload": rag_result, "diagnostic": scores}

        # Otherwise, handle as normal chat -- replace with your chat LLM call
        chat_response = self._chat_reply_stub(message)
        self.append_assistant(chat_response)
        return {"type": "chat", "payload": {"reply": chat_response}, "diagnostic": scores}

    def _chat_reply_stub(self, message: str) -> str:
        # Replace with actual LLM-based chat reply. Use context_summary() to include
        # relevant chat history for the model prompt.
        ctx = self.context_summary()
        return f"(chat) I heard: '{message[:200]}' -- context:\n{ctx[:400]}"


# --------------------------- Example CLI ----------------------------------

def interactive_cli():
    print("RAG-enabled agent (demo). Type 'exit' or Ctrl-C to quit.\n")
    agent = ConversationAgent()

    while True:
        try:
            user = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break
        if not user:
            continue
        if user.strip().lower() in ("exit", "quit"):
            print("bye")
            break

        out = agent.get_reply(user)
        if out["type"] == "rag":
            payload = out["payload"]
            print("\n--- RAG ANSWER ---")
            print(payload["answer"])
            if payload.get("citations"):
                print("\nSources:")
                for s in payload["citations"]:
                    print(" -", s)
            print("------------------\n")
        else:
            print("\n", out["payload"]["reply"], "\n")


# --------------------------- If run as script -----------------------------

if __name__ == "__main__":
    interactive_cli()

