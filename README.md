# AutoMeetingMinutes

Auto Meeting Minutes is a tool based on OpenAI/AzureOpenAI to chat with video/audio and output the meeting minute with build-in prompt.
![Sample Screenshots](/img/Screenshot_page.png)

## How to use

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```
2. Run the Streamlit application:

```bash
streamlit run .\AutoMeetingMinutes_webui.py
```

## Setup OpenAI and Azure OpenAI
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