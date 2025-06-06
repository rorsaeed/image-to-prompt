# Image-to-Prompt AI Assistant

This Python application provides a user-friendly interface to interact with local Large Language Models (LLMs) for generating text prompts from images. It supports both **Ollama** and **LM Studio** as backends.

![Screenshot Placeholder](https://github.com/user-attachments/assets/f3775242-51e7-4192-8653-0d5b0fa1b3bf)

## Features

- **Dual API Support**: Works with both Ollama and LM Studio APIs.
- **Multi-Model Interaction**: Select and query multiple models simultaneously.
- **Image-to-Prompt**: Upload images via file selector or drag-and-drop to generate descriptive prompts.
- **Model Management**: Unload models from memory (Ollama only) to free up VRAM.
- **Custom System Prompts**: Define, save, and reuse custom system prompts.
- **Conversation History**: View and export full conversations as `.txt` or `.json` files.
- **User-Friendly UI**: Includes copy-to-clipboard buttons, loading indicators, and error handling.
- **Persistent Settings**: Remembers your last selected model and API settings.
- **Light/Dark Theme**: Supported via Streamlit's built-in theme settings ( hamburger menu -> Settings).

## ?? Prerequisites

Before running the application, you **MUST** have one of the following installed and running:

1.  **[Ollama](https://ollama.com/)**:
    -   Install Ollama on your system.
    -   Pull a vision-capable model, for example: `ollama run llava`.
    -   Ensure the Ollama service is running in the background.

2.  **[LM Studio](https://lmstudio.ai/)**:
    -   Install LM Studio.
    -   From the home screen, search for and download a vision model (e.g., a `LLaVA` or `Moondream` GGUF).
    -   Go to the **"Local Server"** tab (the `<->` icon).
    -   Select your downloaded model from the dropdown at the top.
    -   Click **"Start Server"**.

**You must have a vision-capable model ready to use.**

## Installation & Usage

1.  **Navigate into the project directory:**
    ```bash
    cd image-to-prompt-app
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

5.  Your web browser will open with the application running. Configure the API provider, URL, and select your model(s) from the sidebar to begin.
