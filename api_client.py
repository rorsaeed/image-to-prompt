import requests
import json
import base64
from pathlib import Path
import subprocess
import mimetypes
import time

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
    
    def upload_video_to_google(self, video_path):
        """Upload a video file to Google Files API and return the file URI."""
        if self.provider != "Google" or not self.google_api_key:
            raise ValueError("Video upload is only supported for Google provider with API key")
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(video_path)
        if not mime_type or not mime_type.startswith('video/'):
            raise ValueError(f"Invalid video file type: {mime_type}")
        
        # Step 1: Start resumable upload
        upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={self.google_api_key}"
        
        metadata = {
            "file": {
                "display_name": Path(video_path).name
            }
        }
        
        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(Path(video_path).stat().st_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json"
        }
        
        response = requests.post(upload_url, headers=headers, json=metadata)
        response.raise_for_status()
        
        upload_session_url = response.headers.get("X-Goog-Upload-URL")
        if not upload_session_url:
            raise ValueError("Failed to get upload session URL")
        
        # Step 2: Upload the file content
        with open(video_path, "rb") as video_file:
            video_content = video_file.read()
        
        upload_headers = {
            "Content-Length": str(len(video_content)),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize"
        }
        
        upload_response = requests.post(upload_session_url, headers=upload_headers, data=video_content)
        upload_response.raise_for_status()
        
        file_info = upload_response.json()
        file_uri = file_info.get("file", {}).get("uri")
        
        if not file_uri:
            raise ValueError("Failed to get file URI from upload response")
        
        # Step 3: Wait for processing to complete
        self._wait_for_file_processing(file_uri)
        
        return file_uri
    
    def _wait_for_file_processing(self, file_uri):
        """Wait for the uploaded file to be processed by Google."""
        file_id = file_uri.split("/")[-1]
        get_url = f"https://generativelanguage.googleapis.com/v1beta/files/{file_id}?key={self.google_api_key}"
        
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = requests.get(get_url)
            response.raise_for_status()
            
            file_data = response.json()
            state = file_data.get("state")
            
            if state == "ACTIVE":
                return  # File is ready
            elif state == "FAILED":
                raise ValueError(f"File processing failed: {file_data.get('error', 'Unknown error')}")
            
            time.sleep(2)  # Wait 2 seconds before checking again
        
        raise ValueError("File processing timed out")

    def generate_chat_response(self, model, messages, images=None, videos=None, stream=True):
        """Sends a request to the chat API and yields the response chunks."""
        headers = {"Content-Type": "application/json"}
        
        if self.provider == "Google":
            if not self.google_api_key:
                yield "--- \n**API Key Error:**\n\n`Google API key is not set.`"
                return

            # --- Google API Payload Translation ---
            # 1. Separate system prompt and regular messages
            system_prompt = ""
            regular_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    system_prompt = msg.get('content', '')
                else:
                    regular_messages.append(msg)

            # 2. Prepend system prompt to the first user message
            if system_prompt and regular_messages and regular_messages[0]['role'] == 'user':
                regular_messages[0]['content'] = f"{system_prompt}\n\n{regular_messages[0]['content']}"
            
            # 3. Merge consecutive messages and handle images
            merged_contents = []
            current_role = None
            current_parts = []

            for msg in regular_messages:
                role = "user" if msg['role'] == "user" else "model"
                
                # If role changes, finalize the previous entry
                if role != current_role and current_role is not None:
                    merged_contents.append({"role": current_role, "parts": current_parts})
                    current_parts = []
                
                current_role = role
                
                # Add text part (validate content is not empty)
                content = msg.get('content', '').strip()
                if content:
                    current_parts.append({"text": content})

                # Add image parts (only for the last user message)
                if msg['role'] == 'user' and images:
                    for img_path in images:
                        # Proper MIME type detection with fallback
                        file_ext = img_path.suffix.lower().lstrip('.')
                        mime_type_map = {
                            'jpg': 'image/jpeg',
                            'jpeg': 'image/jpeg', 
                            'png': 'image/png',
                            'gif': 'image/gif',
                            'webp': 'image/webp',
                            'bmp': 'image/bmp'
                        }
                        mime_type = mime_type_map.get(file_ext, 'image/jpeg')
                        
                        try:
                            image_data = self._encode_image(img_path)
                            if image_data:  # Only add if encoding was successful
                                current_parts.append({
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": image_data
                                    }
                                })
                        except Exception as e:
                            print(f"Warning: Failed to encode image {img_path}: {e}")
                            continue
                    images = None # Process images only once
                
                # Add video parts (only for the last user message)
                if msg['role'] == 'user' and videos:
                    for video_path in videos:
                        try:
                            # Upload video to Google Files API and get URI
                            file_uri = self.upload_video_to_google(video_path)
                            current_parts.append({
                                "file_data": {
                                    "mime_type": mimetypes.guess_type(video_path)[0],
                                    "file_uri": file_uri
                                }
                            })
                        except Exception as e:
                            # If video upload fails, add an error message
                            current_parts.append({
                                "text": f"[Video upload failed: {str(e)}]"
                            })
                    videos = None # Process videos only once

            # Add the last pending message
            if current_role and current_parts:
                merged_contents.append({"role": current_role, "parts": current_parts})

            # 4. Ensure the conversation starts with a 'user' role
            if merged_contents and merged_contents[0]['role'] != 'user':
                 # If the first message is from the model, add a dummy user message
                merged_contents.insert(0, {"role": "user", "parts": [{"text": "(Previous context)"}]})
            
            # 5. Validate that we have valid content to send
            if not merged_contents:
                yield "--- \n**API Error:**\n\n`No valid content to send to the API.`"
                return
            
            # Ensure all messages have at least one part
            for content in merged_contents:
                if not content.get('parts') or len(content['parts']) == 0:
                    yield "--- \n**API Error:**\n\n`Invalid message structure: empty parts detected.`"
                    return

            payload = {
                "contents": merged_contents,
                "generationConfig": {
                    "temperature": 0.9,
                    "topK": 1,
                    "topP": 1,
                    "maxOutputTokens": 8192, # Increased token limit
                    "stopSequences": []
                }
            }
            
            params = {"key": self.google_api_key}
            if stream:
                params['alt'] = 'sse'

            raw_response_for_debugging = ""
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
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
                                            # Check for safety ratings and blocked prompts
                                            if chunk.get('promptFeedback', {}).get('blockReason'):
                                                reason = chunk['promptFeedback']['blockReason']
                                                yield f"\n--- \n**Content Moderation Error:**\n\n`The request was blocked due to: {reason}`"
                                                return
                                            if not chunk.get('candidates'):
                                                continue # Skip empty chunks

                                            content = chunk['candidates'][0]['content']['parts'][0]['text']
                                            if content:
                                                yield content
                                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                                            # This can happen with malformed SSE data or unexpected structures
                                            print(f"SSE parsing error: {e}\nLine: {json_str}")
                                            continue 
                        else: # Not streaming
                            data = response.json()
                            if data.get('promptFeedback', {}).get('blockReason'):
                                reason = data['promptFeedback']['blockReason']
                                yield f"\n--- \n**Content Moderation Error:**\n\n`The request was blocked due to: {reason}`"
                                return
                            content = data['candidates'][0]['content']['parts'][0]['text']
                            if content:
                                yield content
                    return  # Success, exit retry loop

                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 400 and attempt < max_retries - 1:
                        # Retry on 400 errors (might be transient)
                        print(f"Attempt {attempt + 1} failed with 400 error, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        # Final attempt or non-retryable error
                        error_detail = ""
                        try:
                            error_response = e.response.json()
                            error_detail = error_response.get('error', {}).get('message', str(e))
                        except:
                            error_detail = str(e)
                        yield f"--- \n**API Connection Error:**\n\n`{error_detail}`"
                        return
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed with connection error, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        yield f"--- \n**API Connection Error:**\n\n`{e}`"
                        return
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
                                    if 'choices' in chunk and chunk['choices']:
                                        content = chunk['choices'][0]['delta'].get('content', '')
                                        if content:
                                            yield content
                                    elif 'error' in chunk:
                                        error_message = chunk.get('error', {}).get('message', json.dumps(chunk))
                                        yield f"\n--- \n**API Error:**\n\n`{error_message}`"

                                except json.JSONDecodeError:
                                    # In case of malformed JSON, just continue
                                    continue
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
                                try:
                                    chunk = json.loads(json_str)
                                    if self.provider == "LM Studio":
                                        if 'choices' in chunk and chunk['choices']:
                                            content = chunk['choices'][0]['delta'].get('content', '')
                                        else:
                                            continue
                                    else: # Ollama
                                        content = chunk.get('message', {}).get('content', '')
                                    if content:
                                        yield content
                                except json.JSONDecodeError:
                                    continue
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