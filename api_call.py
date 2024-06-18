import requests
import json
from utils import process_sample_no_stream
from ws import browser_func
from api import tts, download_audio
import time


def speech2text(file_path):
    url = "http://0.0.0.0:9191/speech2text"

    payload = {}
    files=[
    ('file',('recorded_audio.wav',open(file_path,'rb'),'audio/wav'))
    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    print(response.text)

    return json.loads(response.text)["transcript"]

def speech2text_vinai(file_path):
    url = "http://103.171.92.194:9191/speech2text"

    payload = {}
    files=[
    ('file',('recorded_audio.wav',open(file_path,'rb'),'audio/wav'))
    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    return json.loads(response.text)["transcript"]


def call_classify_intent_subintent(query):
    PROMPT = """
    You are an expert on intent classification, your task is to take the input query from user and categorize it into one category from categories below.
We have 3 intent including "chat", "search", "create_image" . For "search" intent we have 5 subintent "search", "scroll up", "scroll down", "back", "close" and "choose link forward". I need you to categorize the user query into intent and subintent.

The output format should like this 

{"intent":"search","subintent":"scroll up"}

If the intent is "chat" or "create_image" just leave the subintent as "". If the intent is "search" and the subintent is "search" add the "query" key is the extracted object that user want to query. If the subintent is "choose link forward" add the "index" key is the index of the link that user want to navigate to.

User query:"""
    prompt = PROMPT + query
    out = process_sample_no_stream(prompt)
    out = out[out.find("{"):]
    out = out[:out.find("}")+1]
    print(out)
    out = json.loads(out)
    return out

    
def create_image_api(text):
    url = "http://0.0.0.0:9191/create_image"

    payload = {'text': text}
    files=[

    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    return json.loads(response.text)["image"]

# search_tool = browser_func.Search()

def search_browser(subintent, addition=""):
    global search_tool
    if search_tool is None:
        search_tool = browser_func.Search()
    if subintent == 'search':
        search_tool.search(addition)
    elif subintent == 'scroll up':
        search_tool.scroll_up()
    elif subintent == 'scroll down':
        search_tool.scroll_down()
    elif subintent == 'back':
        search_tool.back()
    elif subintent == 'choose link forward':
        search_tool.navigate(addition)
    else:
        search_tool.driver.quit()
        del search_tool
        search_tool = None
        
def text2speech(text):
    print("text", text)
    url = tts(text)
    print("url",url)
    time.sleep(1)
    download_audio(url,"audio_response.wav")
    

# text2speech("Xin chào, tôi là trợ lý ảo của bạn. Tôi có thể giúp gì cho bạn không?")
