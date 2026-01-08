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
from metadata_extractor import ImageMetadataExtractor

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
		"Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is...\" or similar phrasing.",
		"Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is...\" or similar phrasing.",
		"Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is...\" or similar phrasing.",
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
    if "uploaded_videos" not in st.session_state: st.session_state.uploaded_videos = []
    if "metadata_extractor" not in st.session_state: st.session_state.metadata_extractor = ImageMetadataExtractor()
    
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

def save_uploaded_video(uploaded_file):
    temp_dir = Path("temp_videos"); temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / f"{uuid.uuid4()}_{uploaded_file.name}"
    original_name = uploaded_file.name
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    return (file_path, original_name)

def is_video_file(filename):
    """Check if the file is a supported video format."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
    return Path(filename).suffix.lower() in video_extensions

def extract_ai_prompts_from_metadata(metadata):
    """Extract prompt and negative prompt from AI metadata."""
    if not metadata:
        return None, None
    
    prompt = None
    negative_prompt = None
    
    # Function to parse structured prompt data (like from AUTOMATIC1111)
    def parse_structured_prompt(text_data):
        nonlocal prompt, negative_prompt
        if 'Negative prompt:' in text_data:
            # Split on "Negative prompt:" to separate positive and negative
            parts = text_data.split('Negative prompt:', 1)
            if len(parts) == 2:
                # Extract positive prompt (everything before "Negative prompt:")
                positive_part = parts[0].strip()
                # Remove any leading labels like "Prompt:" or similar
                if positive_part.startswith(('Prompt:', 'prompt:')):
                    positive_part = positive_part.split(':', 1)[1].strip()
                if not prompt and positive_part:
                    prompt = positive_part
                
                # Extract negative prompt (everything after "Negative prompt:" until next parameter)
                negative_part = parts[1].strip()
                # Find where parameters start (usually indicated by newline + parameter name)
                param_indicators = ['\nSteps:', '\nSampler:', '\nCFG scale:', '\nSeed:', '\nSize:', '\nModel:', '\nClip skip:']
                for indicator in param_indicators:
                    if indicator in negative_part:
                        negative_part = negative_part.split(indicator)[0].strip()
                        break
                if not negative_prompt and negative_part:
                    negative_prompt = negative_part
        elif not prompt and len(text_data) > 20:  # If no structured format, use as prompt
            prompt = text_data
    
    # Priority 1: Check PNG text data (most common for AI images)
    if metadata.get('png_text'):
        png_data = metadata['png_text']
        # Check for parameters field (AUTOMATIC1111 format)
        if 'parameters' in png_data:
            parse_structured_prompt(str(png_data['parameters']))
        
        # Check for direct prompt fields
        if not prompt and 'prompt' in png_data:
            prompt = str(png_data['prompt'])
        if not negative_prompt and 'negative_prompt' in png_data:
            negative_prompt = str(png_data['negative_prompt'])
    
    # Priority 2: Check AI metadata if not found in PNG text
    if (not prompt or not negative_prompt) and metadata.get('ai_metadata'):
        ai_data = metadata['ai_metadata']
        for key, value in ai_data.items():
            key_lower = key.lower()
            if 'prompt' in key_lower and 'negative' not in key_lower and not prompt:
                prompt = str(value)
            elif 'negative' in key_lower and 'prompt' in key_lower and not negative_prompt:
                negative_prompt = str(value)
            elif key_lower in ['user_comment', 'ai prompt/parameters'] and (not prompt or not negative_prompt):
                parse_structured_prompt(str(value))
    
    # Priority 3: Check Windows properties if still not found
    if (not prompt or not negative_prompt) and metadata.get('windows_properties'):
        for key, value in metadata['windows_properties'].items():
            if value and len(str(value)) > 20:
                parse_structured_prompt(str(value))
                if prompt and negative_prompt:  # Stop if we found both
                    break
    
    return prompt, negative_prompt


def display_image_metadata(image_path, original_name):
    """Display image metadata in an expandable section."""
    # Check if the file is a PNG - only extract metadata for PNG files
    file_extension = Path(image_path).suffix.lower()
    if file_extension not in ['.png']:
        # For non-PNG files, show a simple message
        with st.expander(f"📊 View Metadata - {original_name}", expanded=False):
            st.info(f"Metadata extraction is only supported for PNG files. This is a {file_extension.upper()} file.")
        return
    
    metadata = st.session_state.metadata_extractor.extract_metadata(image_path)
    
    if metadata:
        formatted_sections = st.session_state.metadata_extractor.format_metadata_for_display(metadata)
        
        if formatted_sections:
            with st.expander(f"📊 View Metadata - {original_name}", expanded=False):
                for idx, section in enumerate(formatted_sections):
                    # Determine CSS class based on section title
                    css_class = "metadata-section"
                    if "AI Generation" in section['title']:
                        css_class += " ai-metadata"
                    elif "Technical" in section['title']:
                        css_class += " technical-metadata"
                    elif "File Information" in section['title']:
                        css_class += " file-metadata"
                    
                    # Create a styled container for each section
                    st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metadata-title">{section["title"]}</div>', unsafe_allow_html=True)
                    
                    # Display content in a nice format
                    for key, value in section['content'].items():
                        # Handle long text values (like prompts)
                        if len(str(value)) > 100:
                            st.markdown(f'<div class="metadata-item"><span class="metadata-key">{key}:</span></div>', unsafe_allow_html=True)
                            st.code(str(value), language=None)
                        else:
                            # Truncate very long values for display
                            display_value = str(value)
                            if len(display_value) > 50:
                                display_value = display_value[:47] + "..."
                            st.markdown(f'<div class="metadata-item"><span class="metadata-key">{key}:</span> <span class="metadata-value">{html.escape(display_value)}</span></div>', unsafe_allow_html=True)
                    

                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.write("")  # Add some spacing
        else:
            with st.expander(f"📊 View Metadata - {original_name}", expanded=False):
                st.info("No AI generation metadata found in this image.")
    else:
        with st.expander(f"📊 View Metadata - {original_name}", expanded=False):
            st.error("Could not extract metadata from this image.")

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
    st.session_state.uploaded_videos = []; st.session_state.uploader_key = str(uuid.uuid4())

def load_chat_callback():
    selected_chat_file = st.session_state.get("selected_chat")
    if selected_chat_file:
        filepath = cm.CONVERSATIONS_DIR / selected_chat_file
        st.session_state.messages = cm.load_conversation(filepath)
        st.session_state.chat_id = selected_chat_file
        st.session_state.uploaded_files = []; st.session_state.uploaded_videos = []
        st.session_state.uploader_key = str(uuid.uuid4())

# <<< The `regenerate_last_response` function has been REMOVED >>>
def run_generation_logic():
    try:
        last_user_message = st.session_state.messages[-1]
        image_info = last_user_message.get("images", [])
        image_paths = [Path(info["path"]) for info in image_info]
        video_info = last_user_message.get("videos", [])
        video_paths = [Path(info["path"]) for info in video_info]
        api_messages = [{"role": "system", "content": st.session_state.current_system_prompt}]
        for msg in st.session_state.messages:
            if msg['role'] != 'system': api_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Create APIClient with current provider settings
        current_provider_name = st.session_state.config["api_provider"]
        provider_config = st.session_state.config["providers"].get(current_provider_name, {})
        
        minicpm_config = None
        if current_provider_name == "MiniCPM":
            minicpm_config = provider_config
        
        api_client = APIClient(
            provider=current_provider_name,
            base_url=provider_config.get("api_base_url") if current_provider_name not in ["Google", "MiniCPM"] else None,
            google_api_key=st.session_state.config.get("google_api_key") if current_provider_name == "Google" else None,
            ollama_keep_alive=provider_config.get("keep_alive") if current_provider_name == "Ollama" else None,
            unload_after_response=provider_config.get("unload_after_response", False) if current_provider_name == "LM Studio" else provider_config.get("auto_unload", False) if current_provider_name == "MiniCPM" else False,
            minicpm_config=minicpm_config
        )
        
        for model in st.session_state.config["providers"][st.session_state.config["api_provider"]]["selected_models"]:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                prefix = f"**Response from `{model}`:**\n\n"
                message_placeholder.markdown(prefix + "▌")
                
                full_response = ""
                try:
                    messages_for_this_model = copy.deepcopy(api_messages)
                    stream_generator = api_client.generate_chat_response(model=model, messages=messages_for_this_model, images=image_paths, videos=video_paths)
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
        st.session_state.generating = False
        # <<< CHANGE: The lines that cleared uploaded_files and reset the uploader_key have been REMOVED >>>
        # This makes the uploaded images persist for re-analysis.
        st.rerun()

def process_and_send_message(prompt_text, uploaded_file_info, uploaded_video_info=None):
    current_provider_name = st.session_state.config.get("api_provider", "Ollama") # Get current provider
    selected_models = st.session_state.config["providers"].get(current_provider_name, {}).get("selected_models", [])
    if not selected_models: st.error("Please select at least one model from the sidebar."); return

    image_info_for_message = [{"path": str(info[0]), "name": info[1]} for info in uploaded_file_info]
    video_info_for_message = [{"path": str(info[0]), "name": info[1]} for info in uploaded_video_info] if uploaded_video_info else []
    
    has_media = bool(image_info_for_message or video_info_for_message)
    is_media_only_request = not prompt_text.strip()
    
    if is_media_only_request and has_media:
        if video_info_for_message:
            internal_prompt = "Analyze the attached video(s) and image(s) according to the system prompt." if image_info_for_message else "Analyze the attached video(s) according to the system prompt."
        else:
            internal_prompt = "Analyze the attached image(s) according to the system prompt."
    else:
        internal_prompt = prompt_text
    
    display_text = prompt_text
    user_message = {"role": "user", "content": internal_prompt, "display_content": display_text, "id": str(uuid.uuid4())}
    if image_info_for_message: user_message["images"] = image_info_for_message
    if video_info_for_message: user_message["videos"] = video_info_for_message
    
    st.session_state.messages.append(user_message)
    auto_save_chat()
    st.session_state.generating = True
    st.rerun()

def remove_uploaded_image(idx):
    if 0 <= idx < len(st.session_state.uploaded_files):
        del st.session_state.uploaded_files[idx]
        st.rerun()

def remove_uploaded_video(idx):
    if 0 <= idx < len(st.session_state.uploaded_videos):
        del st.session_state.uploaded_videos[idx]
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
        model = msg.get('model', st.session_state.config['providers'][st.session_state.config['api_provider']]['selected_models'][0])
        # Prepare MiniCPM config if needed
        minicpm_config = None
        if st.session_state.config['api_provider'] == 'MiniCPM':
            minicpm_config = st.session_state.config['providers'][st.session_state.config['api_provider']]
        
        api_client = APIClient(
            provider=st.session_state.config['api_provider'], 
            base_url=st.session_state.config['providers'][st.session_state.config['api_provider']]['api_base_url'] if st.session_state.config['api_provider'] not in ['Google', 'MiniCPM'] else None, 
            google_api_key=st.session_state.config.get('google_api_key') if st.session_state.config['api_provider'] == 'Google' else None,
            unload_after_response=st.session_state.config['providers'][st.session_state.config['api_provider']].get("unload_after_response", False) if st.session_state.config['api_provider'] in ['LM Studio', 'MiniCPM'] else False,
            minicpm_config=minicpm_config
        )
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
    api_providers = ["Ollama", "LM Studio", "Koboldcpp", "Google", "MiniCPM"]
    current_provider = st.session_state.config.get("api_provider", "Ollama")
    st.session_state.config["api_provider"] = st.radio(
        "API Provider",
        api_providers,
        index=api_providers.index(current_provider) if current_provider in api_providers else 0,
        key="api_provider_selector",
        disabled=st.session_state.generating,
        on_change=lambda: cm.save_config(st.session_state.config) # Save config on provider change
    )

    # Get current provider's specific config
    current_provider_name = st.session_state.config["api_provider"]
    provider_config = st.session_state.config["providers"].setdefault(current_provider_name, {"api_base_url": "", "selected_models": []})

    if current_provider_name == "Google":
        st.session_state.config["google_api_key"] = st.text_input(
            "Google API Key",
            value=st.session_state.config.get("google_api_key", ""),
            key="google_api_key_input",
            type="password",
            disabled=st.session_state.generating,
            on_change=lambda: cm.save_config(st.session_state.config)
        )
    elif current_provider_name == "MiniCPM":
        st.subheader("MiniCPM Configuration")
        
        # Device selection
        device_options = ["auto", "cuda", "cpu"]
        current_device = provider_config.get("device", "auto")
        provider_config["device"] = st.selectbox(
            "Device",
            device_options,
            index=device_options.index(current_device) if current_device in device_options else 0,
            key="minicpm_device_selector",
            disabled=st.session_state.generating,
            on_change=lambda: cm.save_config(st.session_state.config)
        )
        
        # Video Analysis Parameters
        st.subheader("Video Analysis Parameters")
        
        provider_config["max_num_frames"] = st.number_input(
            "MAX_NUM_FRAMES (Total frames to analyze)",
            min_value=1,
            max_value=1000,
            value=provider_config.get("max_num_frames", 180),
            key="minicpm_max_frames",
            disabled=st.session_state.generating,
            help="Controls the total number of frames to analyze from the video",
            on_change=lambda: cm.save_config(st.session_state.config)
        )
        
        provider_config["max_num_packing"] = st.number_input(
            "MAX_NUM_PACKING (Frame grouping)",
            min_value=1,
            max_value=6,
            value=provider_config.get("max_num_packing", 3),
            key="minicpm_max_packing",
            disabled=st.session_state.generating,
            help="Determines how frames are grouped together for processing (valid range: 1-6)",
            on_change=lambda: cm.save_config(st.session_state.config)
        )
        
        provider_config["default_fps"] = st.number_input(
            "Default FPS (Leave 0 for auto-calculation)",
            min_value=0.0,
            max_value=60.0,
            value=float(provider_config.get("default_fps", 3.0)),
            step=0.1,
            key="minicpm_default_fps",
            disabled=st.session_state.generating,
            help="FPS for video sampling. Set to 0 to auto-calculate based on video duration and frame parameters",
            on_change=lambda: cm.save_config(st.session_state.config)
        )
        
        # Additional options
        provider_config["enable_thinking"] = st.checkbox(
            "Enable thinking mode",
            value=provider_config.get("enable_thinking", False),
            key="minicpm_enable_thinking",
            disabled=st.session_state.generating,
            help="Enable thinking mode for more detailed analysis",
            on_change=lambda: cm.save_config(st.session_state.config)
        )
        
        provider_config["auto_unload"] = st.checkbox(
            "Auto-unload model after response",
            value=provider_config.get("auto_unload", False),
            key="minicpm_auto_unload",
            disabled=st.session_state.generating,
            help="Automatically unload the model from memory after each response to save GPU memory",
            on_change=lambda: cm.save_config(st.session_state.config)
        )
    else:
        default_urls = {
            "Ollama": "http://localhost:11434",
            "LM Studio": "http://localhost:1234",
            "Koboldcpp": "http://localhost:5001"
        }
        # Use the saved URL for the current provider, or its default if not set
        current_api_base_url = provider_config.get("api_base_url", default_urls.get(current_provider_name, ""))
        provider_config["api_base_url"] = st.text_input(
            "API Base URL",
            value=current_api_base_url,
            key=f"api_base_url_input_{current_provider_name}", # Unique key for each provider
            disabled=st.session_state.generating,
            on_change=lambda: cm.save_config(st.session_state.config)
        )
        if current_provider_name == "LM Studio":
            provider_config["unload_after_response"] = st.checkbox(
                "Unload model after response",
                value=provider_config.get("unload_after_response", False),
                key=f"unload_after_response_checkbox_{current_provider_name}",
                disabled=st.session_state.generating,
                on_change=lambda: cm.save_config(st.session_state.config)
            )
        if current_provider_name == "Ollama":
            current_keep_alive = provider_config.get("keep_alive", -1) # Default to -1 (server default)
            provider_config["keep_alive"] = st.number_input(
                "Keep Alive (seconds, -1 for server default, 0 for no cache)",
                min_value=-1,
                value=current_keep_alive,
                key=f"ollama_keep_alive_{current_provider_name}",
                disabled=st.session_state.generating,
                on_change=lambda: cm.save_config(st.session_state.config)
            )
    
    # Instantiate APIClient with current provider's settings
    minicpm_config = None
    if current_provider_name == "MiniCPM":
        minicpm_config = provider_config
    
    api_client = APIClient(
        provider=current_provider_name,
        base_url=provider_config.get("api_base_url") if current_provider_name not in ["Google", "MiniCPM"] else None,
        google_api_key=st.session_state.config.get("google_api_key") if current_provider_name == "Google" else None,
        ollama_keep_alive=provider_config.get("keep_alive") if current_provider_name == "Ollama" else None, # Pass keep_alive
        unload_after_response=provider_config.get("unload_after_response", False) if current_provider_name == "LM Studio" else provider_config.get("auto_unload", False) if current_provider_name == "MiniCPM" else False,
        minicpm_config=minicpm_config
    )

    with st.spinner("Fetching available models..."):
        available_models = api_client.get_available_models()

    if not available_models:
        st.error("Could not connect or no models found.")
    else:
        # Use the saved selected models for the current provider
        saved_selection = provider_config.get("selected_models", [])
        valid_selection = [model for model in saved_selection if model in available_models]
        
        # Update the selected_models for the current provider
        provider_config["selected_models"] = st.multiselect(
            "Select Model(s)",
            options=available_models,
            default=valid_selection,
            key=f"selected_models_multiselect_{current_provider_name}", # Unique key
            disabled=st.session_state.generating,
            on_change=lambda: cm.save_config(st.session_state.config)
        )

    st.subheader("Model Management")
    is_ollama = current_provider_name == "Ollama"
    
    
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
    button[data-testid=\"baseButton\"]:has(div:contains('×')),
    button[data-testid=\"baseButton\"]:has(span:contains('×')) {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #222 !important;
        font-size: 1em;
        padding: 0.05em 0.3em;
        margin: 0;
        transition: color 0.2s;
    }
    button[data-testid=\"baseButton\"]:has(div:contains('×')):hover, 
    button[data-testid=\"baseButton\"]:has(span:contains('×')):hover {
        color: #000 !important;
        background: transparent !important;
    }
    
    /* Global video size constraints - apply to all videos */
    [data-testid="stVideo"] {
        max-width: 250px !important;
        width: auto !important;
    }
    
    [data-testid="stVideo"] > div {
        max-width: 250px !important;
        width: auto !important;
    }
    
    [data-testid="stVideo"] video {
        max-width: 250px !important;
        max-height: 200px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
    }
    
    /* Video thumbnail styling with higher specificity */
    .video-thumbnail [data-testid="stVideo"],
    .video-thumbnail [data-testid="stVideo"] > div,
    .video-thumbnail [data-testid="stVideo"] > div > div {
        max-width: 250px !important;
        width: auto !important;
    }
    
    .video-thumbnail [data-testid="stVideo"] > div > div > video,
    .video-thumbnail [data-testid="stVideo"] video {
        max-width: 250px !important;
        max-height: 200px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
    }
    
    .video-thumbnail-small [data-testid="stVideo"],
    .video-thumbnail-small [data-testid="stVideo"] > div,
    .video-thumbnail-small [data-testid="stVideo"] > div > div {
        max-width: 250px !important;
        width: auto !important;
    }
    
    .video-thumbnail-small [data-testid="stVideo"] > div > div > video,
    .video-thumbnail-small [data-testid="stVideo"] video {
        max-width: 250px !important;
        max-height: 150px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
    }
    
    /* Metadata display styling - Using Streamlit default colors */
    .metadata-section {
        background-color: transparent;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 4px solid #007bff;
        border: 1px solid rgba(49, 51, 63, 0.2);
    }
    
    .metadata-title {
        font-weight: bold;
        margin-bottom: 8px;
        font-size: 14px;
    }
    
    .metadata-item {
        margin: 4px 0;
        font-size: 12px;
        line-height: 1.4;
    }
    
    .metadata-key {
        font-weight: 600;
        opacity: 0.7;
    }
    
    .metadata-value {
        word-break: break-word;
    }
    
    .ai-metadata {
        border-left-color: #28a745;
        background-color: rgba(40, 167, 69, 0.05);
    }
    
    .technical-metadata {
        border-left-color: #ffc107;
        background-color: rgba(255, 193, 7, 0.05);
    }
    
    /* Light theme variables (default) */
    :root {
        --background-color: #ffffff;
        --section-background-color: #f9fafb;
        --border-color: #9ca3af;
        --text-color: #111827;
        --secondary-text-color: #1f2937;
        --ai-background-color: #ecfdf5;
        --technical-background-color: #fef3c7;
        --file-background-color: #ede9fe;
    }
    
    /* Dark theme variables - Multiple detection methods */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #2d3748;
            --section-background-color: #2d3748;
            --border-color: #4a5568;
            --text-color: #e2e8f0;
            --secondary-text-color: #a0aec0;
            --ai-background-color: #1a2f1a;
            --technical-background-color: #2d2a1a;
            --file-background-color: #2d1b3d;
        }
    }
    
    /* Streamlit dark theme detection - body class */
    body[data-theme="dark"],
    .stApp[data-theme="dark"],
    [data-theme="dark"] {
        --background-color: #2d3748;
        --section-background-color: #2d3748;
        --border-color: #4a5568;
        --text-color: #e2e8f0;
        --secondary-text-color: #a0aec0;
        --ai-background-color: #1a2f1a;
        --technical-background-color: #2d2a1a;
        --file-background-color: #2d1b3d;
    }
    
    /* Additional Streamlit dark theme selectors */
    .stApp:has([data-testid="stSidebar"][data-theme="dark"]),
    html:has(.stApp[data-theme="dark"]) {
        --background-color: #2d3748;
        --section-background-color: #2d3748;
        --border-color: #4a5568;
        --text-color: #e2e8f0;
        --secondary-text-color: #a0aec0;
        --ai-background-color: #1a2f1a;
        --technical-background-color: #2d2a1a;
        --file-background-color: #2d1b3d;
    }
    
    .file-metadata {
        border-left-color: #6f42c1;
        background-color: rgba(111, 66, 193, 0.05);
    }
    

    </style>
    
    <script>
    // Enhanced theme detection for Streamlit
    function detectAndApplyTheme() {
        const stApp = document.querySelector('.stApp');
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        const body = document.body;
        const html = document.documentElement;
        
        // Check multiple sources for theme
        const isDark = 
            window.matchMedia('(prefers-color-scheme: dark)').matches ||
            (stApp && stApp.getAttribute('data-theme') === 'dark') ||
            (sidebar && sidebar.getAttribute('data-theme') === 'dark') ||
            (body && body.getAttribute('data-theme') === 'dark') ||
            (html && html.getAttribute('data-theme') === 'dark');
        
        // Apply theme to multiple elements
        const themeValue = isDark ? 'dark' : 'light';
        
        if (stApp) stApp.setAttribute('data-theme', themeValue);
        if (body) body.setAttribute('data-theme', themeValue);
        if (html) html.setAttribute('data-theme', themeValue);
        
        console.log('Theme detected and applied:', themeValue);
    }
    
    // Run theme detection
    detectAndApplyTheme();
    
    // Watch for theme changes
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', detectAndApplyTheme);
    }
    
    // Watch for DOM changes (Streamlit updates)
    const observer = new MutationObserver(detectAndApplyTheme);
    observer.observe(document.body, { 
        attributes: true, 
        attributeFilter: ['data-theme', 'class'],
        subtree: true 
    });
    
    // Periodic check as fallback
    setInterval(detectAndApplyTheme, 1000);
    </script>
    """,
    unsafe_allow_html=True
)

# --- Theme Detection and Application ---
# Add additional theme detection using Streamlit's component system
st.markdown("""
<script>
// Force theme detection on page load
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
        detectAndApplyTheme();
    }, 100);
});

// Additional theme detection for Streamlit updates
window.addEventListener('load', function() {
    setTimeout(function() {
        detectAndApplyTheme();
    }, 500);
});
</script>
""", unsafe_allow_html=True)

# --- Main Application Area ---
st.title("🖼️ Image-to-Prompt AI Assistant")
st.warning("**Important:** For local models, ensure **LM Studio** or **Ollama** is running with the API server enabled and a vision model loaded. For Google, ensure you have entered a valid API key.")

tab1, tab2, tab3 = st.tabs(["Chat", "Bulk Analysis", "Recommended Models"])

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
                if "videos" in message:
                    # Create more columns to make videos smaller
                    num_videos = len(message["videos"])
                    # Use more columns than videos to create smaller containers
                    video_cols = st.columns([1, 2, 1] if num_videos == 1 else [2] * num_videos + [1] * max(0, 3 - num_videos))
                    for j, video_info in enumerate(message["videos"]):
                        col_index = 1 if num_videos == 1 else j  # Center single video, otherwise use sequential columns
                        with video_cols[col_index]:
                            video_path = Path(video_info["path"])
                            if video_path.exists():
                                st.markdown('<div class="video-thumbnail-small">', unsafe_allow_html=True)
                                st.video(str(video_path))
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.caption(video_info["name"])
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
        current_provider = st.session_state.config.get("api_provider", "Ollama")
        
        # Image upload section
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
        

        
        # Video upload section (for Google and MiniCPM)
        if current_provider in ["Google", "MiniCPM"]:
            provider_text = "Google & MiniCPM" if current_provider == "MiniCPM" else "Google Only"
            st.subheader(f"Upload Videos ({provider_text})")
            uploaded_videos_from_widget = st.file_uploader(
                "Upload video(s)", 
                type=["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv", "m4v"],
                accept_multiple_files=True,
                key=f"video_{st.session_state.uploader_key}"
            )

            if uploaded_videos_from_widget:
                # Process multiple uploaded video files
                new_video_uploads = [save_uploaded_video(file) for file in uploaded_videos_from_widget]
                st.session_state.uploaded_videos = new_video_uploads
            elif not st.session_state.uploaded_videos:
                # Initialize to empty list if nothing uploaded and no existing files
                st.session_state.uploaded_videos = []
        else:
            # Clear videos if not using Google or MiniCPM
            st.session_state.uploaded_videos = []
        
        # Display all uploaded images in a grid
        if st.session_state.uploaded_files:
            st.write("**Uploaded Images:**")
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
                            
                            # Display metadata below the image
                            display_image_metadata(file_path, original_name)
        
        # Display all uploaded videos in a grid (for Google and MiniCPM)
        if st.session_state.uploaded_videos and current_provider in ["Google", "MiniCPM"]:
            st.write("**Uploaded Videos:**")
            num_videos = len(st.session_state.uploaded_videos)
            cols_per_row = min(4, num_videos)  # Maximum 4 videos per row for smaller thumbnails
            
            # Calculate how many rows we need
            num_rows = (num_videos + cols_per_row - 1) // cols_per_row
            
            # Create a grid to display videos with smaller columns
            for row in range(num_rows):
                # Create columns with specific widths to make videos smaller
                if cols_per_row == 1:
                    cols = st.columns([1, 2, 1])  # Center single video
                    active_cols = [1]
                elif cols_per_row == 2:
                    cols = st.columns([1, 2, 1, 2, 1])  # Two videos with spacing
                    active_cols = [1, 3]
                else:
                    cols = st.columns(cols_per_row + 2)  # Add padding columns
                    active_cols = list(range(1, cols_per_row + 1))
                
                for col_idx in range(cols_per_row):
                    vid_idx = row * cols_per_row + col_idx
                    if vid_idx < num_videos:
                        file_path, original_name = st.session_state.uploaded_videos[vid_idx]
                        with cols[active_cols[col_idx] if col_idx < len(active_cols) else col_idx]:
                            st.markdown('<div class="video-thumbnail">', unsafe_allow_html=True)
                            st.video(str(file_path))
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.caption(original_name)
                            if st.button("×", key=f"remove_vid_{vid_idx}"):
                                remove_uploaded_video(vid_idx)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            has_media = bool(st.session_state.uploaded_files or st.session_state.uploaded_videos)
            if has_media:
                media_text = []
                if st.session_state.uploaded_files:
                    media_text.append("Image(s)")
                if st.session_state.uploaded_videos:
                    media_text.append("Video(s)")
                button_text = f"Analyze {' & '.join(media_text)}"
                if st.button(button_text):
                    process_and_send_message(
                        prompt_text="", 
                        uploaded_file_info=st.session_state.uploaded_files,
                        uploaded_video_info=st.session_state.uploaded_videos
                    )
        with col2:
            # Chat input
            if prompt := st.chat_input("Type your message here..."):
                process_and_send_message(
                    prompt_text=prompt, 
                    uploaded_file_info=st.session_state.uploaded_files,
                    uploaded_video_info=st.session_state.uploaded_videos
                )

with tab2:
    bulk_analysis_page()

with tab3:
    st.header("Recommended Models")

    st.subheader("gemma-3-27b (24Gb+ Vram)")
    st.markdown("""
    - **LM Studio:** [Download](https://model.lmstudio.ai/download/mlabonne/gemma-3-27b-it-abliterated-GGUF)
    - **Ollama:** [gemma3](https://ollama.com/library/gemma3)
      ```bash
      ollama run gemma3:27b
      ```
    """)

    st.subheader("gemma-3-12b (8Gb+ Vram)")
    st.markdown("""
    - **LM Studio:** [Download](https://model.lmstudio.ai/download/mlabonne/gemma-3-12b-it-abliterated-GGUF)
    - **Ollama:** [gemma3](https://ollama.com/library/gemma3)
      ```bash
      ollama run gemma3:12b
      ```
    """)

    st.subheader("gemma-3-4b (4Gb+ Vram)")
    st.markdown("""
    - **LM Studio:** [Download](https://model.lmstudio.ai/download/mlabonne/gemma-3-4b-it-abliterated-GGUF)
    - **Ollama:** [gemma3](https://ollama.com/library/gemma3)
      ```bash
      ollama run gemma3
      ```
    """)

    st.subheader("llama-joycaption-beta-one-hf-llava (12Gb+ Vram)")
    st.markdown("""
    *Best for system prompt builder*
    - **LM Studio:** [Download](https://model.lmstudio.ai/download/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf)
    - **Ollama:** [aha2025/llama-joycaption-beta-one-hf-llava](https://ollama.com/aha2025/llama-joycaption-beta-one-hf-llava)
      ```bash
      ollama run aha2025/llama-joycaption-beta-one-hf-llava
      ```
    """)

    st.subheader("Qwen2.5-VL-7B (8Gb+ Vram)")
    st.markdown("""
    - **LM Studio:** [Download](https://model.lmstudio.ai/download/Misaka27260/Qwen2.5-VL-7B-Instruct-abliterated-GGUF)
    - **Ollama:** [qwen2.5vl](https://ollama.com/library/qwen2.5vl)
      ```bash
      ollama run qwen2.5vl
      ```
    """)

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

# st.markdown("**Prompt:**")
# st.code(prompt_text, language=None)  # Shows a copy button with wrapping
