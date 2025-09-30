# Image-to-Prompt AI Assistant

This Python application provides a user-friendly interface to interact with local Large Language Models (LLMs) for generating text prompts from images or enhancing a simple prompt. It supports popular local API servers like **Ollama, LM Studio, and Koboldcpp**.

![Application Screenshot](https://github.com/rorsaeed/image-to-prompt/blob/main/Screenshot.png)

## Features

- **Multi-API Support**: Works with Ollama, LM Studio, and Koboldcpp APIs, and Google's Gemini API.
- **Multi-Model Interaction**: Select and query multiple models simultaneously.
- **Image-to-Prompt**: Upload images via file selector or drag-and-drop to generate descriptive prompts.
- **Video Analysis**: Upload and analyze video files using Google's Gemini models with Files API integration.
- **Bulk Image Analysis**: Select a folder of images and analyze them all at once, with an option to save the generated prompts to text files.
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

---

## 🎥 Video Support

The application now supports video analysis using **Google's Gemini models**. This feature allows you to upload and analyze video files for content description, scene analysis, and more.

### Requirements for Video Analysis
- **Google API Provider**: Video support is only available when using Google's Gemini models
- **Google API Key**: You must have a valid Google API key configured
- **Supported Formats**: MP4, AVI, MOV, MKV, WebM

### How to Use Video Analysis
1. **Select Google Provider**: In the sidebar, choose "Google" as your API provider
2. **Enter API Key**: Input your Google API key in the configuration section
3. **Choose a Model**: Select a Gemini model (e.g., gemini-1.5-flash, gemini-1.5-pro)
4. **Upload Videos**: Use the video upload section that appears when Google is selected
5. **Analyze**: Add your prompt and click "Analyze Media" to process the video

### Video Processing
- Videos are automatically uploaded to Google's Files API using resumable upload
- The system waits for Google to process the video before analysis
- Videos are temporarily stored in the `temp_videos/` directory
- Error handling ensures graceful fallback if video upload fails

### Supported Video Analysis Tasks
- Content description and summarization
- Scene-by-scene analysis
- Object and activity detection
- Text extraction from video frames
- Audio transcription (if supported by the model)
- Custom analysis based on your prompts



## 🚀 First-Time Windows Setup (Easy Installer)

This is the recommended method for most Windows users. This script will automatically install everything you need.

1.  **Download the Installer:**
    Go to the [**GitHub Releases Page**](https://github.com/rorsaeed/image-to-prompt/releases) and download the `Image-to-Prompt-Installer.zip` file from the latest release.

2.  **Unzip the File:**
    Right-click the downloaded `.zip` file and select "Extract All..." to unzip the folder.

3.  **Run as Administrator:**
    Open the unzipped folder, right-click on the `setup_everything.bat` file, and choose **"Run as administrator"**.


4.  **Approve Permissions:**
    Windows will ask for permission to run the script. Click **"Yes"**. A PowerShell window will open and begin the installation process.

5.  **Wait for Installation:**
    The script will automatically check for and install Git, Python, and LM Studio if they are missing. It will then download the application code and set up all the necessary Python packages. This may take several minutes.

6.  **Follow Final Instructions:**
    Once finished, the script will open the final installation folder (`C:\ImageToPromptApp`) and display a message with the next steps.

---

## ⚙️ How to Use the App

After installation, follow these critical steps:

1.  **Open LM Studio:** Search for it in your Windows Start Menu and open it.
2.  **Download a Vision Model:** Inside LM Studio, use the search bar (🔍) to find and download a vision-capable GGUF model (e.g., search for `mlabonne_gemma-3-27b-it-abliterated-GGUF`).
3.  **Start the Server:**
    -   Go to the **Local Server** tab (`<->`).
    -   At the top, ensure the **Server Preset** is set to **`OpenAI API`**.
    -   Select the model you just downloaded.
    -   Click **"Start Server"**.
4.  **Run the App:**
    -   Go to the application folder (`C:\ImageToPromptApp`).
    -   Double-click the **`run.bat`** file. Your web browser will open with the application ready to use!



## Second installation method if you have all the required applications (for Windows)

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

## Updates
-   **2025-01-08:** Added video upload and analysis support for Google's Gemini models using Files API integration.
-   **2025-09-02:** Added support for Google's Gemini Flash models (1.5, 2.0, and 2.5).
-   **2025-08-31:** Add Advanced System Prompt Builder.
-   **2025-08-30:** Added a "Bulk Analysis" tab to analyze all images in a selected folder. You can now enter a folder path and optionally save the generated prompts to text files in the same directory.
