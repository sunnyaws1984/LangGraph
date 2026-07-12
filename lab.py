"""
A customer writes in with a support message. This graph:
  1. Uses Gemini to CLASSIFY the message (category + urgency).
  2. Routes the ticket based on that classification (conditional edge).
  3. Uses Gemini AGAIN to draft a reply — an urgent escalation note for the
     support team if it's urgent, or a friendly auto-reply if it's not.

This one script deliberately covers every core LangGraph concept you need
to know before building anything bigger:

  - State        -> a shared dict (TypedDict) that flows through the graph
  - Nodes        -> plain functions that read state and return updates
  - LLM in a node -> calling Gemini from inside a node, twice, for two
                     different jobs (classification, then reply-writing)
  - Edges        -> add_edge() for a fixed "always go here next"
  - Conditional edges -> add_conditional_edges() to branch based on state
  - START / END  -> the graph's entry and exit points
"""

import os
import json
from typing import TypedDict, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

loaddotenv = True  # load .env file if present, for GOOGLE_API_KEY
# ---------------------------------------------------------------------------
# 1. STATE — the data that flows through every node in the graph.
# ---------------------------------------------------------------------------
class TicketState(TypedDict):
    customer_message: str   # what the customer wrote in
    category: str            # e.g. "billing", "technical", "general"
    urgency: str             # "high" or "normal"
    reply: str               # the drafted response

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# ---------------------------------------------------------------------------
# 2. NODES — plain functions: (state) -> dict of fields to update.
# ---------------------------------------------------------------------------
def classify_ticket(state: TicketState) -> dict:
    """Node 1: ask Gemini to classify the ticket. We ask for strict JSON
    back so the rest of the graph can rely on structured fields instead of
    parsing free-form text.
    """
    prompt = f"""Classify this customer support message.

Message: "{state['customer_message']}"

Respond with ONLY valid JSON, no markdown fences, in this exact shape:
{{"category": "billing" | "technical" | "general", "urgency": "high" | "normal"}}
"""
    response = llm.invoke(prompt)
    data = json.loads(response.content.strip())

    print(f"[classify_ticket] category={data['category']} urgency={data['urgency']}")
    return {"category": data["category"], "urgency": data["urgency"]}


def route_ticket(state: TicketState) -> Literal["urgent", "normal"]:
    """This is the routing function used by add_conditional_edges.
    It doesn't touch state — it just returns a LABEL describing the decision.
    That label is then translated into an actual node name by the mapping
    dict passed to add_conditional_edges() below.
    """
    return "urgent" if state["urgency"] == "high" else "normal"


def escalate(state: TicketState) -> dict:
    """Node 2a: urgent path — draft an internal escalation note for a human agent."""
    prompt = f"""Write a brief, urgent internal escalation note (2-3 sentences) for
a human support agent about this {state['category']} ticket, so they can jump on
it immediately. Message: "{state['customer_message']}"
"""
    response = llm.invoke(prompt)
    print("[escalate] drafted urgent escalation note")
    return {"reply": response.content.strip()}


def auto_reply(state: TicketState) -> dict:
    """Node 2b: normal path — draft a friendly customer-facing reply directly."""
    prompt = f"""Write a short, friendly customer support reply (2-3 sentences) to
this {state['category']} question. Be warm and helpful.
Message: "{state['customer_message']}"
"""
    response = llm.invoke(prompt)
    print("[auto_reply] drafted customer-facing reply")
    return {"reply": response.content.strip()}


# ---------------------------------------------------------------------------
# 3. BUILD THE GRAPH — add nodes, then wire edges between them.
# ---------------------------------------------------------------------------
builder = StateGraph(TicketState)

builder.add_node("classify_ticket", classify_ticket)
builder.add_node("escalate", escalate)
builder.add_node("auto_reply", auto_reply)

builder.add_edge(START, "classify_ticket")

# Conditional edge: after classifying, branch based on urgency.
# route_ticket returns a LABEL ("urgent" / "normal"); this dict maps each
# label to the actual NODE to run next ("escalate" / "auto_reply").
builder.add_conditional_edges(
    "classify_ticket",
    route_ticket,
    {"urgent": "escalate", "normal": "auto_reply"},
)

builder.add_edge("escalate", END)
builder.add_edge("auto_reply", END)

# 4. Compile into a runnable graph.
graph = builder.compile()

# ---------------------------------------------------------------------------
# VISUALIZE THE GRAPH — quick sanity check that the wiring is correct.
# ---------------------------------------------------------------------------
print(graph.get_graph().draw_ascii())
print()

# ---------------------------------------------------------------------------
# RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print("Set your Gemini API key first, e.g.:")
        print("  export GOOGLE_API_KEY=your_key_here")
        print("Get a free key at https://aistudio.google.com/apikey")
        raise SystemExit(1)

    print("=== LangGraph + Gemini: Customer Support Triage ===\n")

    tickets = [
        "The app crashes every time I try to log in, and I have a client demo in 10 minutes!!",
        #"Hi, just wondering what your refund policy is for annual plans.",
    ]

    for msg in tickets:
        print(f"\n--- Incoming ticket ---\n{msg!r}\n")
        result = graph.invoke(
            {"customer_message": msg, "category": "", "urgency": "", "reply": ""}
        )
        print(f"\n[Category: {result['category']} | Urgency: {result['urgency']}]")
        print(f"Reply:\n{result['reply']}")