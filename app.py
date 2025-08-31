# app.py
import streamlit as st
import os
import json
import uuid
import copy 
from datetime import datetime
from pathlib import Path
import html
import re

# --- Local Imports ---
import config_manager as cm
from api_client import APIClient
from bulk_analyzer import bulk_analysis_page

# --- Constants from joycaption ---
CAPTION_TYPE_MAP = {
	"Descriptive": [
		"Write a detailed description for this image.",
		"Write a detailed description for this image in {word_count} words or less.",
		"Write a {length} detailed description for this image.",
	],
	"Descriptive (Casual)": [
		"Write a descriptive caption for this image in a casual tone.",
		"Write a descriptive caption for this image in a casual tone within {word_count} words.",
		"Write a {length} descriptive caption for this image in a casual tone.",
	],
	"Straightforward": [
		"Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
		"Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
		"Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
	],
	"Stable Diffusion Prompt": [
		"Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
		"Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
		"Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
	],
	"MidJourney": [
		"Write a MidJourney prompt for this image.",
		"Write a MidJourney prompt for this image within {word_count} words.",
		"Write a {length} MidJourney prompt for this image.",
	],
	"Danbooru tag list": [
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
	],
	"e621 tag list": [
		"Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
		"Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
		"Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
	],
	"Rule34 tag list": [
		"Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
		"Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
		"Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
	],
	"Booru-like tag list": [
		"Write a list of Booru-like tags for this image.",
		"Write a list of Booru-like tags for this image within {word_count} words.",
		"Write a {length} list of Booru-like tags for this image.",
	],
	"Art Critic": [
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
	],
	"Product Listing": [
		"Write a caption for this image as though it were a product listing.",
		"Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
		"Write a {length} caption for this image as though it were a product listing.",
	],
	"Social Media Post": [
		"Write a caption for this image as if it were being used for a social media post.",
		"Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
		"Write a {length} caption for this image as if it were being used for a social media post.",
	],
}
NAME_OPTION = "If there is a person/character in the image you must refer to them as {name}."

# --- Page Configuration ---
st.set_page_config(
    page_title="Image-to-Prompt AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
def init_session_state():
    if "messages" not in st.session_state: st.session_state.messages = []
    if "chat_id" not in st.session_state: st.session_state.chat_id = None
    if "config" not in st.session_state: st.session_state.config = cm.load_config()
    if "system_prompts" not in st.session_state: st.session_state.system_prompts = cm.load_system_prompts()
    if "current_system_prompt" not in st.session_state:
        last_prompt_name = st.session_state.config.get("last_system_prompt_name", "Default Image-to-Prompt")
        st.session_state.current_system_prompt = st.session_state.system_prompts.get(last_prompt_name, st.session_state.system_prompts.get("Default Image-to-Prompt", ""))
        st.session_state.current_system_prompt_name = last_prompt_name
    if "uploaded_files" not in st.session_state: st.session_state.uploaded_files = []
    if "unload_after_response" not in st.session_state: st.session_state.unload_after_response = False
    if "uploader_key" not in st.session_state: st.session_state.uploader_key = str(uuid.uuid4())
    if "generating" not in st.session_state:
        st.session_state.generating = False

init_session_state()

def remove_thinking_tags(text):
    """Removes <think>...</think> tags from a string."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

# --- Helper & Chat Management Functions ---
def save_uploaded_file(uploaded_file):
    temp_dir = Path("temp_images"); temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / f"{uuid.uuid4()}_{uploaded_file.name}"
    original_name = uploaded_file.name
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    return (file_path, original_name)

def auto_save_chat():
    if not st.session_state.messages: return
    if st.session_state.chat_id is None:
        first_user_message = next((msg['content'] for msg in st.session_state.messages if msg['role'] == 'user'), 'New Chat')
        safe_title = "".join(c for c in first_user_message if c.isalnum() or c in " ._").rstrip()[:50]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        st.session_state.chat_id = f"{safe_title}_{timestamp}.json"
    cm.save_conversation(st.session_state.chat_id, st.session_state.messages)

def start_new_chat():
    st.session_state.messages = []; st.session_state.chat_id = None; st.session_state.uploaded_files = []
    st.session_state.uploader_key = str(uuid.uuid4())

def load_chat_callback():
    selected_chat_file = st.session_state.get("selected_chat")
    if selected_chat_file:
        filepath = cm.CONVERSATIONS_DIR / selected_chat_file
        st.session_state.messages = cm.load_conversation(filepath)
        st.session_state.chat_id = selected_chat_file
        st.session_state.uploaded_files = []; st.session_state.uploader_key = str(uuid.uuid4())

# <<< The `regenerate_last_response` function has been REMOVED >>>
def run_generation_logic():
    try:
        last_user_message = st.session_state.messages[-1]
        image_info = last_user_message.get("images", [])
        image_paths = [Path(info["path"]) for info in image_info]
        api_messages = [{"role": "system", "content": st.session_state.current_system_prompt}]
        for msg in st.session_state.messages:
            if msg['role'] != 'system': api_messages.append({"role": msg["role"], "content": msg["content"]})
        
        for model in st.session_state.config["selected_models"]:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                prefix = f"**Response from `{model}`:**\n\n"
                message_placeholder.markdown(prefix + "▌")
                
                full_response = ""
                try:
                    messages_for_this_model = copy.deepcopy(api_messages)
                    stream_generator = api_client.generate_chat_response(model=model, messages=messages_for_this_model, images=image_paths)
                    for chunk in stream_generator:
                        full_response += chunk
                        # Filter out thinking tags before displaying
                        cleaned_response = remove_thinking_tags(full_response)
                        message_placeholder.markdown(prefix + cleaned_response + "▌")
                    # Final cleanup before saving
                    full_response = remove_thinking_tags(full_response)
                    message_placeholder.markdown(prefix + full_response)
                except Exception as e:
                    st.error(f"An error occurred with model {model}: {e}"); full_response = f"Error: {e}"
                    message_placeholder.markdown(prefix + full_response)
                
                display_response = prefix + full_response
                assistant_message = {"role": "assistant", "content": full_response, "display_content": display_response, "model": model, "id": str(uuid.uuid4())}
                st.session_state.messages.append(assistant_message)
                auto_save_chat()

    finally:
        if st.session_state.get("unload_after_response", False) and st.session_state.config["api_provider"] == "Ollama":
            with st.spinner("Unloading models from memory..."):
                for model in st.session_state.config["selected_models"]: api_client.unload_model(model)
            st.toast("Models unloaded from memory.", icon="✅")
        
        st.session_state.generating = False
        # <<< CHANGE: The lines that cleared uploaded_files and reset the uploader_key have been REMOVED >>>
        # This makes the uploaded images persist for re-analysis.
        st.rerun()

def process_and_send_message(prompt_text, uploaded_file_info):
    if not st.session_state.config["selected_models"]: st.error("Please select at least one model from the sidebar."); return

    image_info_for_message = [{"path": str(info[0]), "name": info[1]} for info in uploaded_file_info]
    is_image_only_request = not prompt_text.strip()
    internal_prompt = prompt_text if not is_image_only_request else "Analyze the attached image(s) according to the system prompt."
    display_text = prompt_text
    user_message = {"role": "user", "content": internal_prompt, "display_content": display_text, "id": str(uuid.uuid4())}
    if image_info_for_message: user_message["images"] = image_info_for_message
    
    st.session_state.messages.append(user_message)
    auto_save_chat()
    st.session_state.generating = True
    st.rerun()

def remove_uploaded_image(idx):
    if 0 <= idx < len(st.session_state.uploaded_files):
        del st.session_state.uploaded_files[idx]
        st.rerun()

def remove_message(idx):
    if 0 <= idx < len(st.session_state.messages):
        del st.session_state.messages[idx]
        st.rerun()

def regenerate_message(idx, container=None):
    # Only allow regeneration for assistant messages
    if 0 <= idx < len(st.session_state.messages):
        msg = st.session_state.messages[idx]
        if msg.get('role') != 'assistant':
            st.warning('Only assistant messages can be regenerated.')
            return
        # Find the user message before this assistant message
        user_idx = idx - 1
        while user_idx >= 0 and st.session_state.messages[user_idx].get('role') != 'user':
            user_idx -= 1
        if user_idx < 0:
            st.warning('No user message found to regenerate from.')
            return
        user_msg = st.session_state.messages[user_idx]
        # Prepare context up to and including the user message
        context_msgs = []
        for m in st.session_state.messages[:user_idx+1]:
            if m.get('role') == 'system':
                continue
            context_msgs.append({'role': m['role'], 'content': m['content']})
        # Add system prompt
        api_messages = [{'role': 'system', 'content': st.session_state.current_system_prompt}] + context_msgs
        # Get images from user message if any
        image_info = user_msg.get('images', [])
        image_paths = [Path(info['path']) for info in image_info]
        # Call the LLM for each selected model (regenerate only for the model of this message)
        model = msg.get('model', st.session_state.config['selected_models'][0])
        api_client = APIClient(provider=st.session_state.config.get('api_provider', 'Ollama'), base_url=st.session_state.config.get('api_base_url', 'http://localhost:11434'))
        target = container if container is not None else st
        message_placeholder = target.empty()
        prefix = f"**Response from `{model}`:**\n\n"
        message_placeholder.markdown(prefix + "▌")
        full_response = ""
        try:
            stream_generator = api_client.generate_chat_response(model=model, messages=copy.deepcopy(api_messages), images=image_paths)
            for chunk in stream_generator:
                full_response += chunk
                # Filter out thinking tags before displaying
                cleaned_response = remove_thinking_tags(full_response)
                message_placeholder.markdown(prefix + cleaned_response + "▌")
            # Final cleanup before saving
            full_response = remove_thinking_tags(full_response)
            message_placeholder.markdown(prefix + full_response)
        except Exception as e:
            st.error(f"An error occurred with model {model}: {e}"); full_response = f"Error: {e}"
            message_placeholder.markdown(prefix + full_response)
        display_response = prefix + full_response
        # Insert a new assistant message right after the current one
        new_message = {
            'role': 'assistant',
            'content': full_response,
            'display_content': display_response,
            'model': model,
            'id': str(uuid.uuid4())
        }
        st.session_state.messages.insert(idx + 1, new_message)
        auto_save_chat()
        st.rerun()

# --- Sidebar ---

with st.sidebar:
    st.header("💬 Conversations")
    if st.button("➕ New Chat", use_container_width=True, on_click=start_new_chat): st.rerun()
    saved_chats = cm.list_conversations()
    chat_options = {f.name: f.name.replace(".json", "").replace("_", " ") for f in saved_chats}
    options_with_placeholder = {"": "Select a chat..."}; options_with_placeholder.update(chat_options)
    current_selection_key = st.session_state.chat_id if st.session_state.chat_id in chat_options else ""
    st.selectbox("Load Chat", options=list(options_with_placeholder.keys()), format_func=lambda x: options_with_placeholder[x], index=list(options_with_placeholder.keys()).index(current_selection_key), on_change=load_chat_callback, key="selected_chat")
    if st.session_state.chat_id:
        with st.expander("Manage Current Chat"):
            new_chat_name = st.text_input("Rename chat:", value=chat_options.get(st.session_state.chat_id, ""))
            if st.button("Rename", use_container_width=True):
                if new_chat_name and new_chat_name != chat_options.get(st.session_state.chat_id, ""):
                    new_filename = new_chat_name.replace(" ", "_") + ".json"
                    if cm.rename_conversation(st.session_state.chat_id, new_filename): st.session_state.chat_id = new_filename; st.toast("Chat renamed!", icon="✏️"); st.rerun()
                    else: st.error("A chat with this name already exists.")
            if st.button("Delete Chat", type="primary", use_container_width=True):
                cm.delete_conversation(st.session_state.chat_id); start_new_chat(); st.toast("Chat deleted!", icon="🗑️"); st.rerun()
    st.divider()
    st.header("⚙️ Configuration")
    api_providers = ["Ollama", "LM Studio", "Koboldcpp"]
    current_provider = st.session_state.config.get("api_provider", "Ollama")
    st.session_state.config["api_provider"] = st.radio(
        "API Provider",
        api_providers,
        index=api_providers.index(current_provider) if current_provider in api_providers else 0,
        key="api_provider_selector",
        disabled=st.session_state.generating
    )
    if st.session_state.config["api_provider"] == "Ollama":
        default_url = "http://localhost:11434"
    elif st.session_state.config["api_provider"] == "LM Studio":
        default_url = "http://localhost:1234"
    else:  # Koboldcpp
        default_url = "http://localhost:5001"
    st.session_state.config["api_base_url"] = st.text_input(
        "API Base URL",
        value=st.session_state.config.get("api_base_url", default_url),
        key="api_base_url_input",
        disabled=st.session_state.generating
    )
    api_client = APIClient(
        provider=st.session_state.config["api_provider"],
        base_url=st.session_state.config["api_base_url"]
    )
    with st.spinner("Fetching available models..."):
        available_models = api_client.get_available_models()
    if not available_models:
        st.error("Could not connect or no models found.")
    else:
        saved_selection = st.session_state.config.get("selected_models", [])
        valid_selection = [model for model in saved_selection if model in available_models]
        st.session_state.config["selected_models"] = st.multiselect(
            "Select Model(s)",
            options=available_models,
            default=valid_selection,
            disabled=st.session_state.generating
        )
    st.subheader("Model Management")
    is_ollama = st.session_state.config["api_provider"] == "Ollama"
    if st.button("Unload Selected Models", disabled=not is_ollama or st.session_state.generating):
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
        help="(Ollama only)",
        disabled=not is_ollama or st.session_state.generating
    )
    if not is_ollama and st.session_state.unload_after_response:
        st.session_state.unload_after_response = False
    # --- System Prompt Management ---
    st.subheader("System Prompt")

    with st.expander("System Prompt Builder"):
        st.header("System Prompt Builder")

        caption_type = st.selectbox(
            "Caption Type",
            list(CAPTION_TYPE_MAP.keys()),
            key="modal_caption_type"
        )

        caption_length = st.selectbox(
            "Caption Length",
            ["any", "very short", "short", "medium-length", "long", "very long"] + [str(i) for i in range(20, 261, 10)],
            key="modal_caption_length"
        )

        st.markdown("**Extra Options**")
        extra_options_keys = [
            NAME_OPTION,
            "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
            "Include information about lighting.",
            "Include information about camera angle.",
            "Include information about whether there is a watermark or not.",
            "Include information about whether there are JPEG artifacts or not.",
            "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
            "Do NOT include anything sexual; keep it PG.",
            "Do NOT mention the image's resolution.",
            "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
            "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
            "Do NOT mention any text that is in the image.",
            "Specify the depth of field and whether the background is in focus or blurred.",
            "If applicable, mention the likely use of artificial or natural lighting sources.",
            "Do NOT use any ambiguous language.",
            "Include whether the image is sfw, suggestive, or nsfw.",
            "ONLY describe the most important elements of the image.",
            "If it is a work of art, do not include the artist's name or the title of the work.",
            "Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",
            "Use vulgar slang and profanity, such as (but not limited to) \"fucking,\" \"slut,\" \"cock,\" etc.",
            "Do NOT use polite euphemisms—lean into blunt, casual phrasing.",
            "Include information about the ages of any people/characters when applicable.",
            "Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
            "Do not mention the mood/feeling/etc of the image.",
            "Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.).",
            "If there is a watermark, you must mention it.",
            'Your response will be used by a text-to-image model, so avoid useless meta phrases like \“This image shows…\", \“You are looking at...\", etc.',
        ]

        extra_options_state = {}
        for i, option in enumerate(extra_options_keys):
            extra_options_state[option] = st.checkbox(option, key=f"modal_extra_option_{i}")

        name_input = ""
        if extra_options_state[NAME_OPTION]:
            name_input = st.text_input("Person / Character Name", key="modal_name_input")

        def build_prompt(caption_type: str, caption_length: str | int, extra_options: dict, name_input: str) -> str:
            if caption_length == "any":
                map_idx = 0
            elif isinstance(caption_length, str) and caption_length.isdigit():
                map_idx = 1
            else:
                map_idx = 2
            
            prompt = CAPTION_TYPE_MAP[caption_type][map_idx]

            selected_options = [option for option, checked in extra_options.items() if checked]
            if selected_options:
                prompt += " " + " ".join(selected_options)
            
            return prompt.format(
                name=name_input or "{NAME}",
                length=caption_length,
                word_count=caption_length,
            )

        if st.button("Generate and Apply Prompt", use_container_width=True):
            st.session_state.current_system_prompt = build_prompt(caption_type, caption_length, extra_options_state, name_input)
            st.rerun()

    prompt_names = list(st.session_state.system_prompts.keys())
    try:
        current_prompt_index = prompt_names.index(st.session_state.current_system_prompt_name) + 1
    except (ValueError, AttributeError):
        current_prompt_index = 0

    def on_prompt_change():
        selected_name = st.session_state.prompt_selector
        if selected_name != "New Custom Prompt":
            st.session_state.current_system_prompt_name = selected_name
            st.session_state.current_system_prompt = st.session_state.system_prompts[selected_name]
            st.session_state.config['last_system_prompt_name'] = selected_name
        else:
            st.session_state.current_system_prompt_name = ""

    st.selectbox("Choose or create a prompt", options=["New Custom Prompt"] + prompt_names, index=current_prompt_index, on_change=on_prompt_change, key="prompt_selector", disabled=st.session_state.generating)
    st.session_state.current_system_prompt = st.text_area("System Prompt Content", value=st.session_state.current_system_prompt, height=200, key="system_prompt_text_area", disabled=st.session_state.generating)
    prompt_save_name = st.text_input("Enter name to save prompt:", value=st.session_state.get("current_system_prompt_name", ""), disabled=st.session_state.generating)
    if st.button("Save System Prompt", disabled=st.session_state.generating):
        if prompt_save_name:
            st.session_state.system_prompts[prompt_save_name] = st.session_state.current_system_prompt
            cm.save_system_prompts(st.session_state.system_prompts)
            st.session_state.current_system_prompt_name = prompt_save_name
            st.session_state.config['last_system_prompt_name'] = prompt_save_name
            st.toast(f"Prompt '{prompt_save_name}' saved!", icon="✅"); st.rerun()
        else: st.warning("Please enter a name for the prompt before saving.")
    cm.save_config(st.session_state.config)
    # --- Export Conversation ---
    st.subheader("Export Conversation")
    if st.session_state.messages:
        col1, col2 = st.columns(2)
        chat_name = st.session_state.chat_id.replace(".json", "") if st.session_state.chat_id else f"conversation_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        txt_data = "".join(f"--- {msg['role'].upper()} ---\n{msg.get('display_content', '')}\n\n" for msg in st.session_state.messages)
        col1.download_button("to .txt", txt_data, f"{chat_name}.txt", "text/plain")
        json_data = json.dumps(st.session_state.messages, indent=2)
        col2.download_button("to .json", json_data, f"{chat_name}.json", "application/json")
    st.sidebar.divider(); st.sidebar.markdown("- [My Website](https://eng.webphotogallery.store/i2p)\n- [GitHub Project Page](https://github.com/rorsaeed/image-to-prompt)")

# --- Custom CSS for transparent X buttons ---
st.markdown(
    """
    <style>
    button[data-testid="baseButton"]:has(div:contains('×')),
    button[data-testid="baseButton"]:has(span:contains('×')) {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #222 !important;
        font-size: 1em;
        padding: 0.05em 0.3em;
        margin: 0;
        transition: color 0.2s;
    }
    button[data-testid="baseButton"]:has(div:contains('×')):hover, 
    button[data-testid="baseButton"]:has(span:contains('×')):hover {
        color: #000 !important;
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Main Application Area ---
st.title("🖼️ Image-to-Prompt AI Assistant")
st.warning("**Important:** Ensure **LM Studio** or **Ollama** is running with the API server enabled and a vision model loaded.")

tab1, tab2 = st.tabs(["Chat", "Bulk Analysis"])

with tab1:
    # Create a list of containers for each message so we can update them in-place
    message_containers = []
    for idx, message in enumerate(st.session_state.messages):
        container = st.container()
        message_containers.append(container)
        with container:
            col_msg, col_btn, col_regen = st.columns([8, 1, 1])
            with col_msg:
                if "display_content" in message:
                    st.markdown(message["display_content"])
                if "images" in message:
                    img_cols = st.columns(len(message["images"]))
                    for j, image_info in enumerate(message["images"]):
                        with img_cols[j]:
                            img_path = Path(image_info["path"])
                            if img_path.exists():
                                st.image(str(img_path), width=150)
                                with st.popover("View Full Size", use_container_width=True):
                                    st.image(str(img_path))
                                st.caption(image_info["name"])
            with col_btn:
                if st.button("×", key=f"remove_msg_{idx}", help="Delete this message"):
                    remove_message(idx)
            with col_regen:
                if message.get('role') == 'assistant':
                    if st.button("↻", key=f"regen_msg_{idx}", help="Regenerate this message"):
                        regenerate_message(idx, message_containers[idx])

    if st.session_state.generating:
        run_generation_logic()
    else:
        st.subheader("Upload Images (Optional)")
        uploaded_files_from_widget = st.file_uploader(
            "Upload image(s)", 
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,  # Allow multiple images
            key=st.session_state.uploader_key
        )

        if uploaded_files_from_widget:
            # Process multiple uploaded files
            new_uploads = [save_uploaded_file(file) for file in uploaded_files_from_widget]
            st.session_state.uploaded_files = new_uploads
        elif not st.session_state.uploaded_files:
            # Initialize to empty list if nothing uploaded and no existing files
            st.session_state.uploaded_files = []
        
        # Display all uploaded images in a grid
        if st.session_state.uploaded_files:
            num_images = len(st.session_state.uploaded_files)
            cols_per_row = min(4, num_images)  # Maximum 4 images per row
            
            # Calculate how many rows we need
            num_rows = (num_images + cols_per_row - 1) // cols_per_row
            
            # Create a grid to display images
            for row in range(num_rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    img_idx = row * cols_per_row + col_idx
                    if img_idx < num_images:
                        file_path, original_name = st.session_state.uploaded_files[img_idx]
                        with cols[col_idx]:
                            st.image(str(file_path), caption=original_name, width=150)
                            if st.button("×", key=f"remove_img_{img_idx}"):
                                remove_uploaded_image(img_idx)
                            with st.popover("View Full Size", use_container_width=True):
                                st.image(str(file_path))
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.session_state.uploaded_files:
                if st.button("Analyze Image(s)"):
                    process_and_send_message(prompt_text="", uploaded_file_info=st.session_state.uploaded_files)
        with col2:
            if prompt := st.chat_input("...or add a message and press Enter"):
                process_and_send_message(prompt_text=prompt, uploaded_file_info=st.session_state.uploaded_files)

with tab2:
    bulk_analysis_page()

# --- CSS for code block wrapping ---
st.markdown("""
    <style>
    /* Target code blocks inside Streamlit */
    .stCode > div {
        overflow-x: auto !important;
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
    .stCode code {
        white-space: pre-wrap !important;
        word-break: break-break-word !important;
    }
    </style>
    """, unsafe_allow_html=True
)

# Example: get your prompt text as before
prompt_text = st.session_state.messages[-1]["content"] if st.session_state.messages else ""

st.markdown("**Prompt:**")
st.code(prompt_text, language=None)  # Shows a copy button with wrapping