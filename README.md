# LangGraph + Gemini POC

A basic LangGraph proof-of-concept: a customer support triage assistant.

Gemini classifies an incoming support message (category + urgency), then
routes it — urgent tickets get an internal escalation note, normal ones get
an auto-drafted customer reply.

## Setup

```bash
pip install langgraph langchain-google-genai grandalf
or pip install requirements.txt
```

```bash
export GOOGLE_API_KEY=your_key_here
```

## Run

```bash
python3 _lab.py
```

## What it demonstrates

- **State** — a shared dict (`TypedDict`) carrying the ticket as it flows through the graph
- **Nodes** — plain functions that call Gemini and update state
- **Conditional edges** — routing to different nodes based on urgency
- **START / END** — the graph's entry and exit points

## Example output

```
--- Incoming ticket ---
'The app crashes every time I try to log in, and I have a client demo in 10 minutes!!'
[classify_ticket] category=technical urgency=high
[escalate] drafted urgent escalation note

[Category: technical | Urgency: high]
Reply:
URGENT ESCALATION: User's app is crashing on login, preventing access for
a client demo starting in 10 minutes. Please jump on this ticket immediately.
```