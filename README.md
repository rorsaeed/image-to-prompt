# Image-to-Prompt AI Assistant

This Python application provides a user-friendly interface to interact with local Large Language Models (LLMs) for generating text prompts from images or enhancing a simple prompt. It supports popular local API servers like **Ollama, LM Studio, and Koboldcpp**.

![Application Screenshot](https://github.com/rorsaeed/image-to-prompt/blob/main/Screenshot.png)

## Features

- **Multi-API Support**: Works with Ollama, LM Studio, and Koboldcpp APIs.
- **Multi-Model Interaction**: Select and query multiple models simultaneously.
- **Image-to-Prompt**: Upload images via file selector or drag-and-drop to generate descriptive prompts.
- **Model Management**: Unload models from memory (Ollama only) to free up VRAM.
- **Custom System Prompts**: Define, save, and reuse custom system prompts, with the last one remembered.
- **Conversation History**: Automatically saves your chats. Load, rename, or delete previous conversations.
- **User-Friendly UI**: Includes clickable image thumbnails, a simple regeneration workflow, and error handling.
- **Persistent Settings**: Remembers your last selected models and system prompt.
- **Light/Dark Theme**: Supported via Streamlit's built-in theme settings (hamburger menu -> Settings).

## Prerequisites

Before running the application, you **MUST** have one of the following installed and running with a vision-capable model loaded.

---

### 1. **[Ollama](https://ollama.com/)**
-   Install Ollama on your system.
-   Pull a vision-capable model, for example: `ollama run llava`.
-   Ensure the Ollama service is running in the background.

---

### 2. **[LM Studio](https://lmstudio.ai/)**
-   Install LM Studio.
-   From the home screen, search for and download a vision model (e.g., a `LLaVA` or `Moondream` GGUF).
-   Go to the **"Local Server"** tab (the `<->` icon).
-   Select your downloaded model from the dropdown at the top.
-   Click **"Start Server"**.

---

### 3. **[Koboldcpp](https://github.com/LostRuins/koboldcpp)**
-   Download the latest `koboldcpp.exe` from their [GitHub Releases](https://github.com/LostRuins/koboldcpp/releases).
-   Download a vision-capable GGUF model (the same kind used by LM Studio).
-   Launch Koboldcpp from your terminal, pointing it to your model. For GPU acceleration, add the appropriate flags (e.g., `--usevulkan`).
    ```bash
    # Example command to launch Koboldcpp
    koboldcpp.exe --model "C:\path\to\your-vision-model.GGUF" --usevulkan
    ```
-   **In the app's sidebar:**
    -   Select the **"LM Studio"** provider (as it uses the same OpenAI-compatible API).
    -   Set the **API Base URL** to `http://localhost:5001`.

**You must have a vision-capable model ready to use.**

#### Recommended Model (for LM Studio / Koboldcpp)
A great GGUF model for generating detailed prompts is available here:
[bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF](https://huggingface.co/bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF)

## Installation & Usage (for Windows)

1.  **Get the code from GitHub:**
    ```bash
    git clone https://github.com/rorsaeed/image-to-prompt.git
    cd image-to-prompt
    ```

2.  **Run the simple installer:**
    Double-click `install.bat`. This will create a Python virtual environment and install all dependencies.

3.  **Launch the app!**
    Double-click `run.bat`.

4.  **(Optional) Update the app:**
    Double-click `update.bat` to pull the latest changes from GitHub.

## Manual Installation

1.  **Get the code from GitHub:**
    ```bash
    git clone https://github.com/rorsaeed/image-to-prompt.git
    cd image-to-prompt
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

Your web browser will open with the application running. Configure the API provider, URL, and select your model(s) from the sidebar to begin.
