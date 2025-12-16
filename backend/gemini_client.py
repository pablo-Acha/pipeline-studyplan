import os
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()


def perform_web_search(query: str, max_results: int = 6) -> List[Dict[str, str]]:
    """Perform a DuckDuckGo search and return a list of results.

    Each result contains: title, href, body.
    """
    results: List[Dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                if not isinstance(result, dict):
                    continue

                title = result.get("title") or ""
                href = result.get("href") or ""
                body = result.get("body") or ""

                if title and href:
                    results.append({
                        "title": title,
                        "href": href,
                        "body": body,
                    })
        return results
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return []


class GeminiClient:
    def __init__(self):
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found")

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.5-flash")
            self.chat = self.model.start_chat(history=[])


        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            self.chat = None

    def _extract_text(self, response) -> str:
        """Safely extract text from Gemini response."""
        if hasattr(response, "text") and response.text:
            return response.text

        try:
            return response.candidates[0].content.parts[0].text
        except Exception:
            return ""

    def generate_response(self, user_input: str) -> str:
        """Generate an AI response with optional web search when prefixed."""
        if not self.chat:
            return "AI service is not configured correctly."

        try:
            text = (user_input or "").strip()
            lower = text.lower()

            # Search trigger
            search_query = None
            if lower.startswith("search:"):
                search_query = text.split(":", 1)[1].strip()
            elif lower.startswith("/search "):
                search_query = text.split(" ", 1)[1].strip()

            if search_query:
                web_results = perform_web_search(search_query, max_results=6)
                if not web_results:
                    return "I could not retrieve web results right now. Please try again."

                refs_lines = []
                for idx, item in enumerate(web_results, start=1):
                    refs_lines.append(
                        f"[{idx}] {item['title']} — {item['href']}\n{item['body']}"
                    )

                refs_block = "\n\n".join(refs_lines)

                system_prompt = (
                    "You are an AI research assistant. Use the provided web search results "
                    "to answer the user query. Synthesize concisely, cite sources inline "
                    "like [1], [2] where relevant, and include a brief summary."
                )

                composed = (
                    f"<system>\n{system_prompt}\n</system>\n"
                    f"<user_query>\n{search_query}\n</user_query>\n"
                    f"<web_results>\n{refs_block}\n</web_results>"
                )

                response = self.chat.send_message(composed)
                text_out = self._extract_text(response)

                return text_out or "I couldn't generate a response from the web results."

            # Default: normal chat
            response = self.chat.send_message(text)
            text_out = self._extract_text(response)

            return text_out or "I couldn't generate a response. Please try again."

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error processing your request."
