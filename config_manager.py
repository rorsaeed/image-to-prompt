# config_manager.py
import json
import os
from pathlib import Path

# --- Configuration Paths ---
DATA_DIR = Path("data")
CONFIG_PATH = DATA_DIR / "config.json"
PROMPTS_PATH = DATA_DIR / "system_prompts.json"
CONVERSATIONS_DIR = DATA_DIR / "conversations"

# --- Predefined System Prompts ---
PREDEFINED_PROMPTS = {
    "Default Image-to-Prompt": (
        "You are an expert at analyzing images and creating detailed, "
        "imaginative, and effective text-to-image prompts. Based on the "
        "image(s) provided, generate a prompt that could be used by an "
        "AI art generator (like Midjourney or DALL-E) to create a similar "
        "or inspired image. Describe the style, subject, composition, lighting, "
        "and any other relevant details. Be creative."
    ),
    "Simple Description": "Describe the contents of the image in a clear and concise manner.",
    "Technical Analysis": (
        "Analyze the provided image from a technical perspective. Describe the "
        "potential camera settings (aperture, shutter speed, ISO), lens type, "
        "lighting setup, and composition techniques used to capture this photo."
    ),
    "Story Starter": "Use the image as inspiration to write the beginning of a short story."
}

def ensure_data_dirs():
    """Create data directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    CONVERSATIONS_DIR.mkdir(exist_ok=True)

def load_config():
    """Loads user configuration from a JSON file."""
    ensure_data_dirs()
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {
        "selected_models": [],
        "api_provider": "Ollama",
        "api_base_url": "http://localhost:11434"
    }

def save_config(config):
    """Saves user configuration to a JSON file."""
    ensure_data_dirs()
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

def load_system_prompts():
    """Loads custom and predefined system prompts."""
    ensure_data_dirs()
    if PROMPTS_PATH.exists():
        with open(PROMPTS_PATH, 'r') as f:
            custom_prompts = json.load(f)
    else:
        custom_prompts = {}
    
    # Combine predefined with custom, giving custom precedence if names conflict
    return {**PREDEFINED_PROMPTS, **custom_prompts}

def save_system_prompts(prompts):
    """Saves custom system prompts to a JSON file."""
    ensure_data_dirs()
    # Filter out predefined prompts before saving
    custom_prompts = {
        name: content for name, content in prompts.items() 
        if name not in PREDEFINED_PROMPTS
    }
    with open(PROMPTS_PATH, 'w') as f:
        json.dump(custom_prompts, f, indent=2)

def save_conversation(conversation_history, filename):
    """Saves a conversation to a file."""
    ensure_data_dirs()
    path = CONVERSATIONS_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        if filename.endswith(".json"):
            json.dump(conversation_history, f, indent=2)
        else: # .txt
            for msg in conversation_history:
                f.write(f"--- {msg['role'].upper()} ---\n")
                if isinstance(msg['content'], str):
                    f.write(msg['content'] + "\n\n")
                # Handle complex content for vision models
                elif isinstance(msg['content'], list):
                    for item in msg['content']:
                        if item['type'] == 'text':
                            f.write(item['text'] + "\n\n")
            f.write("\n" + "="*40 + "\n")
    return str(path.resolve())