# Jerry's AI Playground

This repo includes the tools based on **OpenAI**, **LangChain** and **Streamlit** to increase working efficiency.
1. Auto Meeting Minutes is a tool to chat with video/audio and output the meeting minutes with build-in prompt.
2. Casual Chat is a ChatGPT like ChatBot.
3. FilesChat is a bot to chat with your own documents, supported format including PDF, PPTX, DOCX, TXT.
4. AI translator is a chatbot to translate every words you input between Chinese and English.
5. CSV Chatbot is a chatbot to analyse tabula data, still in developing...
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

Create a file named `key.txt` and add your Azure OpenAI API key, BING_SUBSCRIPTION_KEY(optional) and AZURE_COGNITIVE_SEARCH_API_KEY(optional).

### config.json (Only For Azure)

Create a `config.json` file with the following configuration:

```json
{
    "CHATGPT_MODEL": "xxxxx",
    "OPENAI_API_BASE": "https://xxxxxx.openai.azure.com/",
    "OPENAI_API_VERSION": "xxxxx",
    "EMBEDDING_MODEL": "xxxxx",
    "EMBEDDING_MODEL_VERSION": "xxxxx",
    "BING_SEARCH_URL": "https://api.bing.microsoft.com/v7.0/search",
    "AZURE_COGNITIVE_SEARCH_SERVICE_NAME": "xxxxxxxxxxxx",
    "AZURE_COGNITIVE_SEARCH_INDEX_NAME": "xxxxxxxxxx"
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

### 3-4 CSV ChatBot
1. Azure OpenAI as server and LangChain used to create a chatbot to create python code based on user's query, and give output.