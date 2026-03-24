# CSAI 422 – Assignment 3  
## Building Conversational Agents with Tool Use and Reasoning Techniques

This project implements a conversational weather assistant using tool calling, reasoning techniques, multi-step workflows, and structured outputs.

---

## Features

The project supports three agent types:

### 1. **Basic Agent**
- Uses weather tools to answer current weather and forecast questions.
- Handles single-step tool calling.

### 2. **Chain-of-Thought Agent**
- Uses weather tools plus a calculator tool.
- Solves comparison and calculation-based weather questions.

### 3. **Advanced Agent**
- Supports safe tool execution with validation and error handling.
- Includes both sequential and parallel tool execution helper functions.
- Produces structured JSON outputs for complex queries (printed in the terminal).
- Uses a manual advanced orchestration fallback to ensure stable execution with the selected model.
- Enables comparison, forecasting, and multi-step reasoning across multiple locations.

---

## Files

- `conversational_agent.py` → main Python implementation  
- `.env` → stores API keys (not included in repository for security)  
- `.env.example` → example environment variables  
- `README.md` → project documentation  

---

## Setup Instructions

### 1. Install dependencies

```bash
python3 -m pip install openai python-dotenv requests
```

---

## Environment Variables

Create a `.env` file with:

```
API_KEY=your_api_key_here
BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-8b-instant
WEATHER_API_KEY=your_weather_api_key_here
```

---

## Example Conversations

### Basic Agent
**User:** What is the weather in Cairo?  
**Assistant:** Returns current weather details using the weather tool.

---

### Chain-of-Thought Agent
**User:** What is the temperature difference between Cairo and London?  
**Assistant:** Retrieves both temperatures and calculates the difference using the calculator tool.

---

### Advanced Agent
**User:** Compare the current weather in Cairo, Riyadh, and London.

**Assistant:**
```
Current weather comparison:
Cairo: 17.3°C, Partly cloudy
London: 11.3°C, Overcast
Riyadh: 20.3°C, Thunderstorm

The warmest city right now is Riyadh.
```

---

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
```

---

## Analysis

### Reasoning Comparison

- **Basic Agent**: Fast and simple but limited to single-step queries.  
- **Chain-of-Thought Agent**: Handles multi-step reasoning using both weather tools and a calculator.  
- **Advanced Agent**: Most powerful, supports structured outputs, multi-location queries, and complex reasoning.  

---

### Performance

- Sequential execution is reliable but slower.  
- Parallel execution improves speed when multiple tool calls are independent.  

---

### Challenges

- Handling tool call failures from the LLM  
- Ensuring correct JSON structure for outputs  
- Managing API errors and invalid responses  

---

### Solutions

- Implemented safe tool execution with validation  
- Added fallback manual workflow for stability  
- Used structured output validation to ensure correctness  

---

## How to Run

```bash
python3 conversational_agent.py
```

Then choose:
```
1 → Basic Agent  
2 → Chain-of-Thought Agent  
3 → Advanced Agent  
``` 
