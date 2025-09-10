import requests
import json
import base64
from pathlib import Path
import subprocess

class APIClient:
    """A client for interacting with local LLM APIs (Ollama, LM Studio, and Koboldcpp)."""
    
    def __init__(self, provider="Ollama", base_url="http://localhost:11434", google_api_key=None, ollama_keep_alive=None, unload_after_response=False):
        self.provider = provider
        self.base_url = base_url.rstrip('/') if base_url else None
        self.google_api_key = google_api_key
        self.ollama_keep_alive = ollama_keep_alive
        self.unload_after_response = unload_after_response
        if self.provider in ("LM Studio", "Koboldcpp"):
            self.api_endpoint = f"{self.base_url}/v1/chat/completions"
            self.models_endpoint = f"{self.base_url}/v1/models"
        elif self.provider == "Google":
            self.api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
            self.models_endpoint = "" # Not used for Google
        else:  # Ollama
            self.api_endpoint = f"{self.base_url}/api/chat"
            self.models_endpoint = f"{self.base_url}/api/tags"

    def get_available_models(self):
        """Fetches the list of available models from the API."""
        if self.provider == "Google":
            return ["gemini-1.5-flash-latest", "gemini-2.0-flash", "gemini-2.5-flash"]
        try:
            response = requests.get(self.models_endpoint, timeout=10)
            response.raise_for_status()
            data = response.json()
            # Only show models if the backend matches the provider
            if self.provider == "LM Studio":
                # If any model is owned by koboldcpp, treat as koboldcpp backend and return []
                if data.get('object') == 'list' and any('owned_by' in m and m['owned_by'] == 'koboldcpp' for m in data.get('data', [])):
                    return []
                return [model['id'] for model in data.get('data', []) if model.get('owned_by', '').lower() != 'koboldcpp']
            elif self.provider == "Koboldcpp":
                return [model['id'] for model in data.get('data', []) if model.get('owned_by', '').lower() == 'koboldcpp' or 'owned_by' not in model]
            elif self.provider == "Ollama":
                # Ollama's /api/tags returns a 'models' key with a list of model objects
                return [model['name'] for model in data.get('models', [])]
            else:
                return []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models: {e}")
            return []

    @staticmethod
    def _encode_image(image_path):
        """Encodes an image file to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_chat_response(self, model, messages, images=None, stream=True):
        """Sends a request to the chat API and yields the response chunks."""
        headers = {"Content-Type": "application/json"}
        
        if self.provider == "Google":
            if not self.google_api_key:
                yield "--- \n**API Key Error:**\n\n`Google API key is not set.`"
                return

            # Translate messages to Google format
            contents = []
            for msg in messages:
                role = "user" if msg['role'] == "user" else "model"
                # Handle the case where the last message has images
                if msg['role'] == 'user' and images:
                    parts = [{"type": "text", "text": msg['content']}]
                    for img_path in images:
                        # In-line data is preferred for Google API
                        mime_type = "image/jpeg" # Assuming jpeg, adjust if needed
                        parts.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": self._encode_image(img_path)
                            }
                        })
                    contents.append({"role": role, "parts": parts})
                    images = None # Ensure images are only processed once
                else:
                    contents.append({"role": role, "parts": [{"text": msg['content']}]})

            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.9,
                    "topK": 1,
                    "topP": 1,
                    "maxOutputTokens": 2048,
                    "stopSequences": []
                }
            }
            
            params = {"key": self.google_api_key}
            if stream:
                params['alt'] = 'sse'

            raw_response_for_debugging = ""
            try:
                with requests.post(f"{self.api_endpoint}/{model}:generateContent", headers=headers, json=payload, params=params, stream=stream, timeout=300) as response:
                    response.raise_for_status()
                    if stream:
                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode('utf-8')
                                if decoded_line.startswith('data: '):
                                    json_str = decoded_line[6:].strip()
                                    try:
                                        chunk = json.loads(json_str)
                                        content = chunk['candidates'][0]['content']['parts'][0]['text']
                                        if content:
                                            yield content
                                    except (json.JSONDecodeError, KeyError, IndexError):
                                        continue # Ignore malformed SSE data
                    else: # Not streaming
                        data = response.json()
                        content = data['candidates'][0]['content']['parts'][0]['text']
                        if content:
                            yield content

            except requests.exceptions.RequestException as e:
                yield f"--- \n**API Connection Error:**\n\n`{e}`"
            except json.JSONDecodeError as e:
                yield (
                    f"--- \n**API Error: Failed to decode the server's response.**\n\n"
                    f"**Python Error:** `{e}`\n\n"
                    f"**Full raw response from server:**\n\n```\n{raw_response_for_debugging or 'Response was empty.'}\n```"
                )
            return

        if images and messages and messages[-1]['role'] == 'user':
            last_message = messages[-1]
            if self.provider in ("LM Studio", "Koboldcpp"):
                content_parts = [{"type": "text", "text": last_message['content']}]
                for img_path in images:
                    b64_img = self._encode_image(img_path)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                    })
                last_message['content'] = content_parts
            else: # Ollama
                last_message['images'] = [self._encode_image(img_path) for img_path in images]

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        if self.provider == "Ollama" and self.ollama_keep_alive is not None:
            payload["keep_alive"] = self.ollama_keep_alive
        
        raw_response_for_debugging = ""
        try:
            with requests.post(self.api_endpoint, headers=headers, json=payload, stream=True, timeout=300) as response:
                response.raise_for_status()
                if self.provider == "Koboldcpp":
                    # Koboldcpp streams SSE lines: each line starts with 'data: '
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            raw_response_for_debugging += decoded_line + '\n'
                            if decoded_line.startswith('data: '):
                                json_str = decoded_line[6:].strip()
                                if json_str == "[DONE]":
                                    break
                                if not json_str:
                                    continue
                                try:
                                    chunk = json.loads(json_str)
                                    content = chunk['choices'][0]['delta'].get('content', '')
                                    if content:
                                        yield content
                                except Exception:
                                    continue
                else:
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            raw_response_for_debugging += decoded_line + '\n'
                            if decoded_line.startswith('data: '):
                                json_str = decoded_line[6:].strip()
                                if json_str == "[DONE]":
                                    break
                                if not json_str:
                                    continue
                                chunk = json.loads(json_str)
                                if self.provider == "LM Studio":
                                    content = chunk['choices'][0]['delta'].get('content', '')
                                else: # Ollama
                                    content = chunk['message'].get('content', '')
                                if content:
                                    yield content
                            elif "{" in decoded_line:
                                chunk = json.loads(decoded_line)
                                content = chunk.get('message', {}).get('content', '')
                                if content:
                                    yield content
        except requests.exceptions.RequestException as e:
            yield f"--- \n**API Connection Error:**\n\n`{e}`"
        except json.JSONDecodeError as e:
            yield (
                f"--- \n**API Error: Failed to decode the server's response.**\n\n"
                f"**Python Error:** `{e}`\n\n"
                f"**Full raw response from server:**\n\n```\n{raw_response_for_debugging or 'Response was empty.'}\n```"
            )
        finally:
            if self.provider == "LM Studio" and self.unload_after_response:
                self.unload_model(model)

    def unload_model(self, model_name):
        """Unloads a model from memory."""
        if self.provider == "LM Studio":
            try:
                # Use subprocess to run the lms command
                print("Attempting to unload LM Studio models...")
                result = subprocess.run(["lms", "unload", "--all"], capture_output=True, text=True, check=True)
                print(f"LM Studio unload stdout: {result.stdout}")
                print(f"LM Studio unload stderr: {result.stderr}")
                return {"status": "success", "message": f"Successfully initiated unload for all models via CLI. Output: {result.stdout}"}
            except FileNotFoundError:
                print("LM Studio 'lms' command not found.")
                return {"status": "error", "message": "The 'lms' command was not found. Make sure LM Studio is installed and its command-line tools are in your system's PATH."}
            except subprocess.CalledProcessError as e:
                print(f"LM Studio unload failed. Stderr: {e.stderr}")
                return {"status": "error", "message": f"Failed to unload models via CLI. Error: {e.stderr}"}
        elif self.provider == "Ollama":
            # Ollama handles unloading automatically. Simulate success.
            return {"status": "success", "message": f"'{model_name}' (Ollama) is managed automatically."}
        else: # Koboldcpp
            return {"status": "Unsupported for Koboldcpp"}