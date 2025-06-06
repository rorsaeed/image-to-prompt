# app.py
import streamlit as st
import os
import json
import uuid
import copy 
from datetime import datetime
from pathlib import Path

# --- Local Imports ---
import config_manager as cm
from api_client import APIClient

# --- Page Configuration ---
st.set_page_config(
    page_title="Image-to-Prompt AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "config" not in st.session_state:
        st.session_state.config = cm.load_config()
    if "system_prompts" not in st.session_state:
        st.session_state.system_prompts = cm.load_system_prompts()
    if "current_system_prompt" not in st.session_state:
        st.session_state.current_system_prompt = st.session_state.system_prompts.get("Default Image-to-Prompt", "")
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "unload_after_response" not in st.session_state:
        st.session_state.unload_after_response = False

init_session_state()

# --- Helper Functions ---
def save_uploaded_file(uploaded_file):
    temp_dir = Path("temp_images")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / f"{uuid.uuid4()}_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# --- Central function to process and send messages ---
def process_and_send_message(prompt_text, image_paths):
    if not st.session_state.config["selected_models"]:
        st.error("Please select at least one model from the sidebar.")
        return

    is_image_only_request = not prompt_text.strip()
    internal_prompt = prompt_text if not is_image_only_request else "Analyze the attached image(s) according to the system prompt."
    display_text = prompt_text
    
    user_message = {
        "role": "user", "content": internal_prompt, 
        "display_content": display_text, "id": str(uuid.uuid4())
    }
    if image_paths:
        user_message["images"] = [str(p) for p in image_paths]
    
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        if display_text:
            st.markdown(display_text)
        if image_paths:
            st.image([str(p) for p in image_paths], width=200)

    base_api_messages = [{"role": "system", "content": st.session_state.current_system_prompt}]
    for msg in st.session_state.messages:
        if msg['role'] != 'system':
             base_api_messages.append({"role": msg["role"], "content": msg["content"]})
    
    for model in st.session_state.config["selected_models"]:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            messages_for_this_model = copy.deepcopy(base_api_messages)

            with st.spinner(f"Asking {model}..."):
                try:
                    stream = api_client.generate_chat_response(
                        model=model, messages=messages_for_this_model, images=image_paths)
                    for chunk in stream:
                        full_response += chunk
                        message_placeholder.markdown(f"**Response from `{model}`:**\n\n" + full_response + " |")
                    message_placeholder.markdown(f"**Response from `{model}`:**\n\n" + full_response)
                except Exception as e:
                    st.error(f"An error occurred with model {model}: {e}")
                    full_response = f"Error: {e}"

            assistant_message = {
                "role": "assistant", "content": full_response,
                "display_content": f"**Response from `{model}`:**\n\n" + full_response,
                "model": model, "id": str(uuid.uuid4())
            }
            st.session_state.messages.append(assistant_message)
            st.button(f"📋 Copy Response", key=f"copy_{assistant_message['id']}", on_click=st.write, args=(full_response,))

    if st.session_state.get("unload_after_response", False) and st.session_state.config["api_provider"] == "Ollama":
        with st.spinner("Unloading models from memory..."):
            models_to_unload = st.session_state.config["selected_models"]
            for model in models_to_unload:
                api_client.unload_model(model)
        st.toast("Models unloaded from memory.", icon="✅")
    
    st.session_state.uploaded_files = []
    st.rerun()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.session_state.config["api_provider"] = st.radio(
        "API Provider", ["Ollama", "LM Studio"],
        index=0 if st.session_state.config["api_provider"] == "Ollama" else 1, key="api_provider_selector")
    
    default_url = "http://localhost:11434" if st.session_state.config["api_provider"] == "Ollama" else "http://localhost:1234"
    st.session_state.config["api_base_url"] = st.text_input(
        "API Base URL", value=st.session_state.config.get("api_base_url", default_url), key="api_base_url_input")

    api_client = APIClient(provider=st.session_state.config["api_provider"], base_url=st.session_state.config["api_base_url"])
    
    with st.spinner("Fetching available models..."):
        available_models = api_client.get_available_models()
    
    if not available_models:
        st.error("Could not connect or no models found. Ensure the service is running.")
    else:
        st.session_state.config["selected_models"] = st.multiselect(
            "Select Model(s)", options=available_models,
            default=st.session_state.config.get("selected_models", []))
    
    st.subheader("Model Management")
    is_ollama = st.session_state.config["api_provider"] == "Ollama"
    
    if st.button("Unload Selected Models", disabled=not is_ollama):
        if not st.session_state.config["selected_models"]:
            st.warning("No models selected to unload.")
        else:
            for model_name in st.session_state.config["selected_models"]:
                with st.spinner(f"Unloading {model_name}..."):
                    result = api_client.unload_model(model_name)
                    st.success(f"Successfully unloaded '{model_name}'.") if result['status'] == 'success' else st.error(f"Failed to unload '{model_name}': {result['message']}")
            st.rerun()

    st.session_state.unload_after_response = st.checkbox(
        "Unload models after response",
        value=st.session_state.unload_after_response,
        help="If checked, models used will be unloaded from VRAM after responding. (Ollama only)",
        disabled=not is_ollama
    )
    if not is_ollama and st.session_state.unload_after_response:
        st.session_state.unload_after_response = False

    st.subheader("System Prompt")
    prompt_names = list(st.session_state.system_prompts.keys())
    selected_prompt_name = st.selectbox("Choose or create a prompt", options=["New Custom Prompt"] + prompt_names)

    if selected_prompt_name != "New Custom Prompt":
        st.session_state.current_system_prompt = st.session_state.system_prompts[selected_prompt_name]
    
    st.session_state.current_system_prompt = st.text_area(
        "System Prompt Content", value=st.session_state.current_system_prompt, height=200, key="system_prompt_text_area")
    
    prompt_save_name = st.text_input("Enter name to save prompt:", value=selected_prompt_name if selected_prompt_name != "New Custom Prompt" else "")
    if st.button("Save System Prompt"):
        if prompt_save_name:
            st.session_state.system_prompts[prompt_save_name] = st.session_state.current_system_prompt
            cm.save_system_prompts(st.session_state.system_prompts)
            st.success(f"Prompt '{prompt_save_name}' saved!")
            st.rerun()
        else:
            st.warning("Please enter a name for the prompt before saving.")

    cm.save_config(st.session_state.config)

    st.subheader("Export Conversation")
    if st.session_state.messages:
        col1, col2 = st.columns(2)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        txt_data = "".join(f"--- {msg['role'].upper()} ---\n{msg.get('display_content', '')}\n\n" for msg in st.session_state.messages)
        col1.download_button("to .txt", txt_data, f"conversation_{now}.txt", "text/plain")
        json_data = json.dumps(st.session_state.messages, indent=2)
        col2.download_button("to .json", json_data, f"conversation_{now}.json", "application/json")

    # --- NEW: About & Links Section ---
    st.sidebar.divider()
    st.sidebar.subheader("About & Links")
    st.sidebar.markdown(
        """
        - [My Website](https://eng.webphotogallery.store/i2p)
        - [GitHub Project Page](https://github.com/rorsaeed/image-to-prompt)
        """
    )
    # --- END OF NEW SECTION ---

# --- Main Application Area (No changes here) ---
st.title("🖼️ Image-to-Prompt AI Assistant")
st.warning(
    "**Important:** Ensure **LM Studio** or **Ollama** is running with the API server enabled and a vision model loaded."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["display_content"]:
            st.markdown(message["display_content"])
        if "images" in message:
            st.image([str(Path(p)) for p in message["images"] if Path(p).exists()], width=200)
        if "display_content" in message and message["display_content"]:
            st.button(f"📋 Copy Text", key=f"copy_{message['id']}", on_click=st.write, args=(message['display_content'],))

uploaded_files = st.file_uploader(
    "Upload Images (Drag & Drop Supported)", type=["png", "jpg", "jpeg", "webp", "gif"],
    accept_multiple_files=True, key="file_uploader")

if uploaded_files:
    if not st.session_state.uploaded_files: 
        st.session_state.uploaded_files = [save_uploaded_file(f) for f in uploaded_files]
    st.image([str(p) for p in st.session_state.uploaded_files], width=100)

col1, col2 = st.columns([1, 4])
with col1:
    if st.session_state.uploaded_files:
        if st.button("Analyze Image(s)"):
            process_and_send_message(prompt_text="", image_paths=st.session_state.uploaded_files)
with col2:
    if prompt := st.chat_input("...or add a message and press Enter"):
        process_and_send_message(prompt_text=prompt, image_paths=st.session_state.uploaded_files)
