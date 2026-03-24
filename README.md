# CSAI 422 – Assignment 3  
## Building Conversational Agents with Tool Use and Reasoning Techniques

This project implements a conversational weather assistant using tool calling, reasoning techniques, multi-step workflows, and structured outputs.

## Features

The project supports three agent types:

1. **Basic Agent**
   - Uses weather tools to answer current weather and forecast questions.
   - Handles single-step tool calling.

2. **Chain-of-Thought Agent**
   - Uses weather tools plus a calculator tool.
   - Solves comparison and calculation-based weather questions.

3. Advanced Agent
- Supports safe tool execution with validation and error handling.
- Includes both sequential and parallel tool execution helper functions.
- Produces structured JSON outputs for complex queries (printed in the terminal).
- Uses a manual advanced orchestration fallback to ensure stable execution with the selected model.
- Enables comparison, forecasting, and multi-step reasoning across multiple locations.

---

## Files

- `conversational_agent.py` → main Python implementation
- `.env` → stores API keys and model settings
- `README.md` → project documentation

## Structured Output Example

Example of a structured response produced by the Advanced Agent:

```json
{
  "query_type": "comparison",
  "locations": ["Cairo", "London"],
  "summary": "Compared current temperatures in Cairo and London.",
  "tool_calls_used": ["get_current_weather", "calculator"],
  "final_answer": "The temperature difference is 6.0°C."
}

---

## Setup Instructions

### 1. Install dependencies

```bash
python3 -m pip install openai python-dotenv requests