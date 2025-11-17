

from langchain.tools import tool
import requests

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    try:
        # Replace this with a real API if needed
        response = requests.get(f"https://wttr.in/{city}?format=3")
        return response.text
    except Exception as e:
        return f"Error fetching weather: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    try:
        # Replace this with a real API if needed
        response = requests.get(f"https://wttr.in/{city}?format=3")
        return response.text
    except Exception as e:
        return f"Error fetching weather: {e}"
