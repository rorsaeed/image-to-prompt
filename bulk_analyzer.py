# bulk_analyzer.py
import streamlit as st
import os
import json
from pathlib import Path
import base64
from api_client import APIClient

def get_image_files(folder_path):
    """Gets all image files from a folder."""
    supported_extensions = [".png", ".jpg", ".jpeg", ".webp"]
    return [f for f in Path(folder_path).iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]

def bulk_analysis_page():
    """Creates the bulk analysis page."""
    st.header("Bulk Image Analysis")

    if "bulk_analysis_results" not in st.session_state:
        st.session_state.bulk_analysis_results = []

    # --- Folder Selection ---
    folder_path = st.text_input("Enter the path to the folder containing images:")

    if folder_path and Path(folder_path).is_dir():
        image_files = get_image_files(folder_path)
        st.write(f"Found {len(image_files)} images in the folder.")

        if image_files:
            # --- Analysis Options ---
            save_prompts = st.checkbox("Save prompts to text file (in the same folder)")
            
            if st.button("Analyze All Images", use_container_width=True):
                current_provider_name = st.session_state.config.get("api_provider", "Ollama")
                provider_config = st.session_state.config["providers"].get(current_provider_name, {})
                selected_models = provider_config.get("selected_models", [])

                if not selected_models:
                    st.error("Please select at least one model from the sidebar configuration.")
                    return

                st.session_state.bulk_analysis_results = []
                
                # Instantiate APIClient with the correct provider-specific settings
                api_client = APIClient(
                    provider=current_provider_name,
                    base_url=provider_config.get("api_base_url") if current_provider_name != "Google" else None,
                    google_api_key=st.session_state.config.get("google_api_key") if current_provider_name == "Google" else None,
                    ollama_keep_alive=provider_config.get("keep_alive") if current_provider_name == "Ollama" else None,
                    unload_after_response=provider_config.get("unload_after_response", False) if current_provider_name == "LM Studio" else False
                )
                
                progress_bar = st.progress(0)
                model_to_use = selected_models[0] # Use the first selected model for bulk analysis

                for i, image_path in enumerate(image_files):
                    with st.spinner(f"Analyzing {image_path.name} using {model_to_use}..."):
                        try:
                            messages = [
                                {"role": "system", "content": st.session_state.current_system_prompt},
                                {"role": "user", "content": "Analyze the attached image according to the system prompt."}
                            ]
                            
                            full_response = ""
                            stream_generator = api_client.generate_chat_response(
                                model=model_to_use,
                                messages=messages,
                                images=[image_path]
                            )
                            for chunk in stream_generator:
                                full_response += chunk

                            try:
                                response_json = json.loads(full_response)
                                if 'error' in response_json:
                                    error_message = response_json.get('error', {}).get('message', full_response)
                                    st.error(f"Error analyzing {image_path.name}: {error_message}")
                                    full_response = ""
                            except json.JSONDecodeError:
                                # Not a JSON error response, proceed as normal
                                pass

                            st.session_state.bulk_analysis_results.append({
                                "image_path": str(image_path),
                                "prompt": full_response
                            })

                            if save_prompts and full_response:
                                prompt_file_path = image_path.with_suffix(".txt")
                                with open(prompt_file_path, "w", encoding="utf-8") as f:
                                    f.write(full_response)

                        except Exception as e:
                            st.error(f"Error analyzing {image_path.name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(image_files))

                st.success("Analysis complete!")

    elif folder_path:
        st.error("The specified path is not a valid directory.")

    # --- Display Results ---
    if st.session_state.bulk_analysis_results:
        st.subheader("Analysis Results")
        
        num_images = len(st.session_state.bulk_analysis_results)
        cols_per_row = min(3, num_images)
        num_rows = (num_images + cols_per_row - 1) // cols_per_row

        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                img_idx = row * cols_per_row + col_idx
                if img_idx < num_images:
                    result = st.session_state.bulk_analysis_results[img_idx]
                    with cols[col_idx]:
                        st.image(result["image_path"], use_container_width=True)
                        st.text_area("Generated Prompt", value=result["prompt"], height=200, key=f"prompt_{img_idx}")