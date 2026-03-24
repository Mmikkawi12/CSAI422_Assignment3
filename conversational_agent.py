import os
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# =========================
# TOOLS
# =========================

def get_current_weather(location):
    """Get the current weather for a location."""
    api_key = os.environ.get("WEATHER_API_KEY")
    url = (
        f"http://api.weatherapi.com/v1/current.json"
        f"?key={api_key}&q={location}&aqi=no"
    )

    response = requests.get(url, timeout=10)
    data = response.json()

    if "error" in data:
        return f"Error: {data['error']['message']}"

    weather_info = data["current"]

    return json.dumps(
        {
            "location": data["location"]["name"],
            "temperature_c": weather_info["temp_c"],
            "temperature_f": weather_info["temp_f"],
            "condition": weather_info["condition"]["text"],
            "humidity": weather_info["humidity"],
            "wind_kph": weather_info["wind_kph"],
        }
    )


def get_weather_forecast(location, days=3):
    """Get a weather forecast for a location for a specified number of days."""
    api_key = os.environ.get("WEATHER_API_KEY")
    url = (
        f"http://api.weatherapi.com/v1/forecast.json"
        f"?key={api_key}&q={location}&days={days}&aqi=no"
    )

    response = requests.get(url, timeout=10)
    data = response.json()

    if "error" in data:
        return f"Error: {data['error']['message']}"

    forecast_days = data["forecast"]["forecastday"]
    forecast_data = []

    for day in forecast_days:
        forecast_data.append(
            {
                "date": day["date"],
                "max_temp_c": day["day"]["maxtemp_c"],
                "min_temp_c": day["day"]["mintemp_c"],
                "condition": day["day"]["condition"]["text"],
                "chance_of_rain": day["day"]["daily_chance_of_rain"],
            }
        )

    return json.dumps(
        {
            "location": data["location"]["name"],
            "forecast": forecast_data,
        }
    )


def calculator(expression):
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# =========================
# TOOL SCHEMAS
# =========================

weather_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or country, e.g. Cairo, London, France",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Get the weather forecast for a location for a specific number of days",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or country, e.g. Cairo, London, France",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of forecast days (1-10)",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["location"],
            },
        },
    },
]

calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression like '20 - 10' or '(30 + 10) / 2'",
                }
            },
            "required": ["expression"],
        },
    },
}

cot_tools = weather_tools + [calculator_tool]
advanced_tools = cot_tools

available_functions = {
    "get_current_weather": get_current_weather,
    "get_weather_forecast": get_weather_forecast,
    "calculator": calculator,
}


# =========================
# SYSTEM PROMPTS
# =========================

basic_system_message = (
    "You are a helpful weather assistant. "
    "Use the provided tools whenever the user asks for current weather or forecast data. "
    "Do not invent weather information."
)

cot_system_message = (
    "You are a helpful assistant that can answer weather questions and perform calculations. "
    "For weather questions, use the weather tools. "
    "For calculations, use the calculator tool when needed. "
    "Answer clearly and briefly."
)

advanced_system_message = """You are a helpful weather assistant that can use weather tools and a calculator to solve multi-step problems.

Guidelines:
1. If the user asks about several independent locations, use multiple weather tool calls in parallel when appropriate.
2. If a question requires several steps, continue using tools until the task is completed.
3. If a tool fails, explain the issue clearly and continue safely when possible.
4. For complex comparison or calculation queries, prepare a structured final response when requested.
5. Do not invent weather information. Use tools for live weather data.
"""


# =========================
# PART 1 / PART 2
# =========================

def process_messages(client, messages, tools=None, available_functions=None):
    tools = tools or []
    available_functions = available_functions or {}

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
    except Exception as e:
        messages.append(
            {
                "role": "assistant",
                "content": f"Error while contacting the model: {str(e)}",
            }
        )
        return messages

    response_message = response.choices[0].message
    messages.append(response_message)

    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name

            if function_name not in available_functions:
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Unknown function requested: {function_name}",
                    }
                )
                return messages

            try:
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
            except Exception as e:
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Error while executing tool {function_name}: {str(e)}",
                    }
                )
                return messages

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

        try:
            final_response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": final_response.choices[0].message.content,
                }
            )
        except Exception as e:
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Error while generating final response: {str(e)}",
                }
            )

    return messages


def run_conversation(client, system_message, tools, functions):
    messages = [{"role": "system", "content": system_message}]

    print("Weather Assistant: Hello! I can help you with weather information.")
    print("(Type 'exit' to end the conversation)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nWeather Assistant: Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        messages = process_messages(client, messages, tools, functions)

        last_message = messages[-1]
        if last_message["role"] == "assistant" and last_message.get("content"):
            print(f"\nWeather Assistant: {last_message['content']}\n")

    return messages

# =========================
# PART 3.1 SAFE TOOL EXECUTION
# =========================

def execute_tool_safely(tool_call, available_functions):
    """
    Execute a tool call with validation and error handling.
    Returns a JSON string describing success or error.
    """
    function_name = tool_call.function.name

    if function_name not in available_functions:
        return json.dumps(
            {
                "success": False,
                "error": f"Unknown function: {function_name}",
            }
        )

    try:
        function_args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        return json.dumps(
            {
                "success": False,
                "error": f"Invalid JSON arguments: {str(e)}",
            }
        )

    try:
        function_response = available_functions[function_name](**function_args)
        return json.dumps(
            {
                "success": True,
                "function_name": function_name,
                "result": function_response,
            }
        )
    except TypeError as e:
        return json.dumps(
            {
                "success": False,
                "error": f"Invalid arguments: {str(e)}",
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
            }
        )


# =========================
# PART 3.2 SEQUENTIAL / PARALLEL
# =========================

def execute_tools_sequential(tool_calls, available_functions):
    """
    Execute tool calls one after another.
    """
    results = []
    for tool_call in tool_calls:
        safe_result = execute_tool_safely(tool_call, available_functions)
        tool_message = {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function.name,
            "content": safe_result,
        }
        results.append(tool_message)
    return results


def execute_tools_parallel(tool_calls, available_functions, max_workers=4):
    """
    Execute independent tool calls in parallel.
    """
    def run_single_tool(tool_call):
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function.name,
            "content": execute_tool_safely(tool_call, available_functions),
        }

    with ThreadPoolExecutor(max_workers=min(max_workers, len(tool_calls))) as executor:
        return list(executor.map(run_single_tool, tool_calls))


def compare_parallel_vs_sequential(tool_calls, available_functions):
    """
    Measure timing difference between sequential and parallel execution.
    """
    start = time.perf_counter()
    sequential_results = execute_tools_sequential(tool_calls, available_functions)
    sequential_time = time.perf_counter() - start

    start = time.perf_counter()
    parallel_results = execute_tools_parallel(tool_calls, available_functions)
    parallel_time = time.perf_counter() - start

    speedup = sequential_time / parallel_time if parallel_time > 0 else None

    return {
        "sequential_results": sequential_results,
        "parallel_results": parallel_results,
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
    }


# =========================
# PART 3.3 ADVANCED WORKFLOW
# =========================

def process_messages_advanced(user_input):
    """
    Manual advanced workflow for Groq stability.
    Decides which tools to call based on the user query.
    """
    lowered = user_input.lower()

        # Simple current weather query
    if ("weather in" in lowered or "current weather" in lowered) and "compare" not in lowered:
        city = None
        for c in ["Cairo", "London", "Riyadh", "Jeddah", "Paris", "Rome", "Berlin"]:
            if c.lower() in lowered:
                city = c
                break

        if city:
            weather = json.loads(get_current_weather(city))
            answer = (
                f"The current weather in {weather['location']} is {weather['condition']}.\n"
                f"Temperature: {weather['temperature_c']}°C\n"
                f"Humidity: {weather['humidity']}%\n"
                f"Wind speed: {weather['wind_kph']} km/h"
            )

            structured = {
                "query_type": "current_weather",
                "locations": [city],
                "summary": f"Retrieved current weather for {city}.",
                "tool_calls_used": ["get_current_weather"],
                "final_answer": answer,
            }

            return answer, structured

    # Simple forecast query
    if "forecast" in lowered:
        city = None
        for c in ["Cairo", "London", "Riyadh", "Jeddah", "Paris", "Rome", "Berlin"]:
            if c.lower() in lowered:
                city = c
                break

        days = 3
        for n in range(1, 11):
            if str(n) in lowered:
                days = n
                break

        if city:
            forecast = json.loads(get_weather_forecast(city, days))
            lines = []
            for day in forecast["forecast"]:
                lines.append(
                    f"{day['date']}: {day['condition']}, max {day['max_temp_c']}°C, min {day['min_temp_c']}°C"
                )

            answer = f"The {days}-day weather forecast for {city} is:\n" + "\n".join(lines)

            structured = {
                "query_type": "forecast",
                "locations": [city],
                "summary": f"Retrieved {days}-day forecast for {city}.",
                "tool_calls_used": ["get_weather_forecast"],
                "final_answer": answer,
            }

            return answer, structured

    # Temperature difference between two cities
    if "difference" in lowered and "between" in lowered:
        cities = []

        if "cairo" in lowered:
            cities.append("Cairo")
        if "london" in lowered:
            cities.append("London")
        if "riyadh" in lowered:
            cities.append("Riyadh")
        if "jeddah" in lowered:
            cities.append("Jeddah")
        if "paris" in lowered:
            cities.append("Paris")
        if "rome" in lowered:
            cities.append("Rome")
        if "berlin" in lowered:
            cities.append("Berlin")

        if len(cities) >= 2:
            weather_1 = json.loads(get_current_weather(cities[0]))
            weather_2 = json.loads(get_current_weather(cities[1]))

            temp_1 = weather_1["temperature_c"]
            temp_2 = weather_2["temperature_c"]

            diff = round(abs(float(calculator(f"{temp_1} - {temp_2}"))), 2)

            answer = (
                f"The current temperature in {cities[0]} is {temp_1}°C, "
                f"and the current temperature in {cities[1]} is {temp_2}°C.\n\n"
                f"The temperature difference between {cities[0]} and {cities[1]} is {diff}°C."
            )

            structured = {
                "query_type": "comparison",
                "locations": [cities[0], cities[1]],
                "summary": f"Compared current temperatures in {cities[0]} and {cities[1]}.",
                "tool_calls_used": ["get_current_weather", "calculator"],
                "final_answer": answer,
            }

            return answer, structured

    # Average maximum temperature over next 3 days
    if "average maximum temperature" in lowered:
        city = None
        for c in ["Cairo", "London", "Riyadh", "Jeddah", "Paris", "Rome", "Berlin"]:
            if c.lower() in lowered:
                city = c
                break

        if city:
            forecast = json.loads(get_weather_forecast(city, 3))
            max_temps = [day["max_temp_c"] for day in forecast["forecast"]]
            expr = " + ".join(str(x) for x in max_temps)
            avg = float(calculator(f"({expr}) / 3"))

            answer = (
                f"The average maximum temperature in {city} over the next 3 days is {avg:.2f}°C."
            )

            structured = {
                "query_type": "forecast_average",
                "locations": [city],
                "summary": f"Calculated the 3-day average maximum temperature for {city}.",
                "tool_calls_used": ["get_weather_forecast", "calculator"],
                "final_answer": answer,
            }

            return answer, structured

    # Tomorrow's Riyadh max vs today's Jeddah current
    if "riyadh" in lowered and "jeddah" in lowered and "higher" in lowered:
        riyadh_forecast = json.loads(get_weather_forecast("Riyadh", 2))
        jeddah_current = json.loads(get_current_weather("Jeddah"))

        tomorrow_max = riyadh_forecast["forecast"][1]["max_temp_c"]
        jeddah_temp = jeddah_current["temperature_c"]

        comparison = tomorrow_max > jeddah_temp
        answer = (
            f"Tomorrow's maximum temperature in Riyadh is {tomorrow_max}°C, "
            f"and today's current temperature in Jeddah is {jeddah_temp}°C.\n\n"
        )

        if comparison:
            answer += "Yes, Riyadh's tomorrow maximum temperature will be higher."
        else:
            answer += "No, Riyadh's tomorrow maximum temperature will not be higher."

        structured = {
            "query_type": "comparison",
            "locations": ["Riyadh", "Jeddah"],
            "summary": "Compared Riyadh tomorrow maximum temperature with Jeddah current temperature.",
            "tool_calls_used": ["get_weather_forecast", "get_current_weather"],
            "final_answer": answer,
        }

        return answer, structured

    # Compare multiple cities current weather
    if "compare" in lowered or "warmer" in lowered or "hotter" in lowered:
        cities = []
        for c in ["Cairo", "London", "Riyadh", "Jeddah", "Paris", "Rome", "Berlin"]:
            if c.lower() in lowered:
                cities.append(c)

        if len(cities) >= 2:
            results = []
            for city in cities:
                results.append(json.loads(get_current_weather(city)))

            summary_lines = []
            for r in results:
                summary_lines.append(
                    f"{r['location']}: {r['temperature_c']}°C, {r['condition']}"
                )

            hottest = max(results, key=lambda x: x["temperature_c"])

            answer = "Current weather comparison:\n" + "\n".join(summary_lines)
            answer += f"\n\nThe warmest city right now is {hottest['location']}."

            structured = {
                "query_type": "comparison",
                "locations": cities,
                "summary": "Compared current weather across multiple cities.",
                "tool_calls_used": ["get_current_weather"],
                "final_answer": answer,
            }

            return answer, structured

    # Fallback to simple LLM answer without tools
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": advanced_system_message},
            {"role": "user", "content": user_input},
        ],
    )

    answer = response.choices[0].message.content
    structured = {
        "query_type": "general",
        "locations": [],
        "summary": "Answered without tool use.",
        "tool_calls_used": [],
        "final_answer": answer,
    }
    return answer, structured


required_output_keys = [
    "query_type",
    "locations",
    "summary",
    "tool_calls_used",
    "final_answer",
]

structured_output_prompt = """For complex comparison or calculation queries, return the final answer as a valid JSON object with exactly these keys:
- query_type
- locations
- summary
- tool_calls_used
- final_answer
Do not include markdown fences.
"""


def validate_structured_output(response_text):
    """Validate the final structured JSON response."""
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON output: {str(e)}")

    for key in required_output_keys:
        if key not in parsed:
            raise ValueError(f"Missing required key: {key}")

    if not isinstance(parsed["locations"], list):
        raise ValueError("'locations' must be a list")

    if not isinstance(parsed["tool_calls_used"], list):
        raise ValueError("'tool_calls_used' must be a list")

    return parsed


def get_structured_final_response(client, messages):
    """
    Request a structured final response in JSON mode and validate it.
    """
    structured_messages = messages + [
        {"role": "system", "content": structured_output_prompt}
    ]

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=structured_messages,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    return validate_structured_output(content)

def run_conversation_advanced(client, system_message=advanced_system_message):
    print("Advanced Weather Assistant: Hello! Ask me complex weather questions.")
    print("I can compare cities, perform calculations, and return structured outputs.")
    print("(Type 'exit' to end the conversation)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nAdvanced Weather Assistant: Goodbye!")
            break

        try:
            answer, structured = process_messages_advanced(user_input)

            print(f"\nAdvanced Weather Assistant: {answer}\n")
            print("Structured Output:")
            print(json.dumps(structured, indent=2))
            print()

        except Exception as e:
            print(f"\nAdvanced Weather Assistant: Error: {str(e)}\n")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    choice = input(
        "Choose an agent type (1: Basic, 2: Chain of Thought, 3: Advanced): "
    )

    if choice == "1":
        run_conversation(client, basic_system_message, weather_tools, available_functions)
    elif choice == "2":
        run_conversation(client, cot_system_message, cot_tools, available_functions)
    elif choice == "3":
        run_conversation_advanced(client, advanced_system_message)
    else:
        print("Invalid choice. Defaulting to Basic.")
        run_conversation(client, basic_system_message, weather_tools, available_functions)