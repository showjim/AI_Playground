# Jerry's AI Playground

This repo includes the tools based on AI to increase working efficiency.
1. Auto Meeting Minutes is a tool based on OpenAI/AzureOpenAI to chat with video/audio and output the meeting minutes with build-in prompt.
2. Casual Chat is a ChatGPT like ChatBot.
3. FilesChat is a bot to chat with your documents.
![Sample Screenshots](/img/home_page.png)

## 1. How to use


1. Install ffmpeg, [ffmpeg official website](https://ffmpeg.org/).

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```
3. Run the Streamlit application:

```bash
streamlit run .\home_page.py
```

## 2. Setup OpenAI and Azure OpenAI
To configure the application, you will need to create a `key.txt` file containing your Azure OpenAI API key and a `config.json` file with your desired settings.

### key.txt

Create a file named `key.txt` and add your Azure OpenAI API key as a single line in the file.

### config.json (Only For Azure)

Create a `config.json` file with the following configuration:

```json
{
    "CHATGPT_MODEL": "xxxxx",
    "OPENAI_API_BASE": "https://xxxxxx.openai.azure.com/",
    "OPENAI_API_VERSION": "xxxxx",
    "EMBEDDING_MODEL": "xxxxx",
    "EMBEDDING_MODEL_VERSION": "xxxxx"
}
```

## 3. How it works
### 3-1 Auto Meeting Minutes
1. Ffmpeg is used to convert video to audio;
2. Faster-Whisper is used to convert audio to subtitle;
3. AgglomerativeClustering from sklearn is used to distinguish the speakers;
4. OpenAI is used to do the meeting minute summary.

### 3-2 Casual Chatbot
1. Azure OpenAI GPT3.5 as the server;
2. LangChain is used to create the conversational Chatbot;
3. Streamlit is used to create the web-ui

### 3-3 Files ChatBot
1. Azure OpenAI as server and LangChain used to create a QA chatbot over documents.
