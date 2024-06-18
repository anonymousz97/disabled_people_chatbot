from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import requests
import json
import os
import urllib.parse
import traceback

app = FastAPI()

import requests

#### INTENT CLASSIFICATION ####
import requests
import json
import urllib.parse
import requests
from IPython.display import Video


def tts(text):
	url = "https://api.zalo.ai/v1/tts/synthesize"
	text = urllib.parse.quote(text)

	payload = 'speaker_id=1&input=' + text
	headers = {
	  'apikey': 'rTvA7aPlSORowI7mkkdPVJDHqzhLdcC9',
	  'Content-Type': 'application/x-www-form-urlencoded'
	} 
	
 
	response = requests.request("POST", url, headers=headers, data=payload)
	res = json.loads(response.text)
	if res['error_code'] == 0:
		return res['data']['url'] 
	return ""

def open_tts(url):

	# Send a GET request to download the file
	response = requests.get(url)

	# Save the MP4 file
	with open('video.mp4', 'wb') as file:
		file.write(response.content)
  
def stt(file_path):
	url = "https://viettelgroup.ai/voice/api/asr/v1/rest/decode_file"

	payload = {}
	files=[
	('file',('recorded_audio.wav',open(file_path,'rb'),'audio/wav'))
	]
	headers = {
	'Accept': '*/*',
	'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7,fr-FR;q=0.6,fr;q=0.5,ja;q=0.4,zh-CN;q=0.3,zh;q=0.2',
	'Origin': 'https://viettelgroup.ai',
	'Sec-Fetch-Dest': 'empty',
	'Sec-Fetch-Mode': 'cors',
	'Sec-Fetch-Site': 'same-origin'
	}

	response = requests.request("POST", url, headers=headers, data=payload, files=files)
	print(json.loads(response.text)[0]['result'])
	try:
		res = json.loads(response.text)[0]['result']['hypotheses'][0]['transcript']
	except:
		res = ""

	return res
  
# Define model 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Example data
data = {
	'intent': ['create_image'] * 8 + ['search'] * 7,
	'text': 
		[
			'Tôi muốn tạo ảnh',
			'vẽ',
			'vẽ ảnh',
			'tao ảnh',
			'muốn tạo ảnh',
			'ảnh mới nhé',
			'anh moi',
			'anh moi su dung',
			'tìm kiếm',
			'tim kiem',
			'tìm kiem',
			'tra thông tin',
			'hỏi',
			'cần tìm kiếm',
			'cần tìm kiếm thông tin'
		]
}

df = pd.DataFrame(data)

def train_model(df):
	# Encode labels
	label_encoder = LabelEncoder()
	df['intent'] = label_encoder.fit_transform(df['intent'])

	# Split data
	X_train, X_test, y_train, y_test = train_test_split(df['text'], df['intent'], test_size=0.1, random_state=42)

	# Vectorize text data
	vectorizer = TfidfVectorizer()
	X_train_vectorized = vectorizer.fit_transform(X_train)
	X_test_vectorized = vectorizer.transform(X_test)

	# Create and train the model
	model = LogisticRegression()
	model.fit(X_train_vectorized, y_train)

	# Evaluate the model
	y_pred = model.predict(X_test_vectorized)
	# print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
	return model, vectorizer, label_encoder

# Function to classify new queries
def classify_intent(text, model, vectorizer, label_encoder):
	text_vectorized = vectorizer.transform([text])
	intent_label = model.predict(text_vectorized)[0]
	intent = label_encoder.inverse_transform([intent_label])[0]
	return intent


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Example data
data = {
	'intent': ['search'] * 7 + ['up'] * 5 + ['down'] * 5 + ['back'] * 6 + ['choose'] * 8,
	'text': 
		[
			'tìm kiếm',
			'tim kiem',
			'tìm kiem',
			'tra thông tin',
			'hỏi',
			'cần tìm kiếm',
			'cần tìm kiếm thông tin',
			'kéo lên',
			'lên',
			'lên trên',
			'len trên',
			'cuộn lên',
			'kéo xuống',
			'xuống',
			'xuống dưới',
			'xuống duoi',
			'cuộn xuống',
			'quay lại',
			'trang trước',
			'bách lại',
			'làm ơn bách lại',
			'rời trang',
			'quay lai',
			'chọn link thứ ',
			'chọn linh thú',
			'chọn linh thứ',
			'bài viết thứ',
			'trỏ vào bài',
			'vào trang',
			'bấm vô',
			'bấm vào'
		]
}

df2 = pd.DataFrame(data)

def train_model_search(df2):

	# Encode labels
	label_encoder = LabelEncoder()
	df2['intent'] = label_encoder.fit_transform(df2['intent'])

	# Split data
	X_train2, X_test2, y_train2, y_test2 = train_test_split(df2['text'], df2['intent'], test_size=0.1, random_state=42)

	# Vectorize text data
	vectorizer = TfidfVectorizer()
	X_train_vectorized = vectorizer.fit_transform(X_train2)
	X_test_vectorized = vectorizer.transform(X_test2)

	# Create and train the model
	model_search = LogisticRegression()
	model_search.fit(X_train_vectorized, y_train2)

	# Evaluate the model
	y_pred2 = model_search.predict(X_test_vectorized)
	# print(y_pred2)
	return model_search, vectorizer, label_encoder
# print(classification_report(y_test2, y_pred2, target_names=label_encoder.classes_))

# Function to classify new queries
def classify_intent_search(text, model_search, vectorizer, label_encoder):
	text_vectorized = vectorizer.transform([text])
	intent_label = model_search.predict(text_vectorized)[0]
	intent = label_encoder.inverse_transform([intent_label])[0]
	return intent


model, vectorizer1, label_encoder1 = train_model(df)
model_search, vectorizer2, label_encoder2 = train_model_search(df2)


### CREATE IMAGE ###

import base64
from io import BytesIO
from PIL import Image

def load_image_from_base64(base64_str: str) -> Image:
	# Decode the base64 string
	image_data = base64.b64decode(base64_str)
	# Convert the bytes to a BytesIO object
	image_bytes = BytesIO(image_data)
	# Open the image using PIL
	image = Image.open(image_bytes)
	return image

def create_image_with_prompt(prompt):

	url = "https://api.edenai.run/v2/image/generation"

	payload = json.dumps({
		"providers": "openai,deepai,stabilityai",
		"text": prompt,
		"resolution": "512x512",
		"num_images": 1
	})
	headers = {
	'authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiY2Q4MjE3MjMtZjBiZi00ZmY0LWE0ZjAtM2IwZjJjYWNlNGE3IiwidHlwZSI6ImFwaV90b2tlbiJ9.Oy54j7epOdCyrW_6nkpMyfkGBWqjMhuALx_7krDfNKA',
	'content-type': 'application/json'
	}

	response = requests.request("POST", url, headers=headers, data=payload)
 
	print(response.text)

	res = json.loads(response.text)
	
	img = load_image_from_base64(res['deepai']['items'][0]['image'])
	
	return img


#### SPEECH <-> TEXT ####


def stt(file_path):
		url = "https://viettelgroup.ai/voice/api/asr/v1/rest/decode_file"

		payload = {}
		files=[
			('file',("sent_file",open(file_path,'rb'),'audio/wav'))
		]
		headers = {
			'Accept': '*/*',
			'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7,fr-FR;q=0.6,fr;q=0.5,ja;q=0.4,zh-CN;q=0.3,zh;q=0.2',
			'Origin': 'https://viettelgroup.ai',
			'Sec-Fetch-Dest': 'empty',
			'Sec-Fetch-Mode': 'cors',
			'Sec-Fetch-Site': 'same-origin'
		}

		response = requests.request("POST", url, headers=headers, data=payload, files=files)
		try:
			res = json.loads(response.text)[0]['result']['hypotheses'][0]['transcript']
		except:
			res = ""

		return res

def tts(text):
	try:
		url = "https://api.zalo.ai/v1/tts/synthesize"
		text = urllib.parse.quote(text)

		payload = f'speaker_id=1&input={text}'
		headers = {
			'apikey': 'rTvA7aPlSORowI7mkkdPVJDHqzhLdcC9',
			'Content-Type': 'application/x-www-form-urlencoded'
		}

		response = requests.post(url, headers=headers, data=payload)
		# response.raise_for_status()  # Raise an error for bad HTTP status codes

		res = json.loads(response.text)
		if res['error_code'] == 0:
			return res['data']['url']
		return ""

	except Exception as e:
		traceback.print_exc()
		return ""

def download_audio(url, filename):
	try:
		response = requests.get(url)
		response.raise_for_status()  # Raise an error for bad HTTP status codes

		with open(filename, 'wb') as file:
			file.write(response.content)

	except Exception as e:
		traceback.print_exc()
  

#### FASTAPI ####
import numpy as np

@app.post("/create_image")
async def create_image2(text: str = Form(...)):
	try:
		image_base64 = create_image_with_prompt(text)
		image_base64 = np.array(image_base64).tolist()
		if not image_base64:
			raise HTTPException(status_code=400, detail="Image generation failed")
		return JSONResponse(content={"image": image_base64})

	except Exception as e:
		traceback.print_exc()
		raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/text2speech")
async def text2speech(text: str = Form(...)):
	try:
		audio_url = tts(text)
		if not audio_url:
			raise HTTPException(status_code=400, detail="Text-to-speech conversion failed")

		audio_file = "audio.mp4"
		download_audio(audio_url, audio_file)

		return FileResponse(audio_file, media_type="audio/mp4", filename="audio.mp4")

	except Exception as e:
		traceback.print_exc()
		raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/speech2text")
async def speech2text(file: UploadFile = File(...)):
	try:
		file_path = f"temp_{file.filename}"
		print(file_path)
		with open(file_path, "wb+") as file_object:
			file_object.write(file.file.read())

		transcript = stt(file_path)
		print(transcript)
		os.remove(file_path)  # Clean up the temporary file

		if not transcript:
			raise HTTPException(status_code=400, detail="Speech-to-text conversion failed")

		return JSONResponse(content={"transcript": transcript})

	except Exception as e:
		traceback.print_exc()
		raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/intent")
async def intent(text: str = Form(...)):
	try:
		predicted_intent = classify_intent(text, model, vectorizer1, label_encoder1)
		return JSONResponse(content={"intent": predicted_intent})

	except Exception as e:
		traceback.print_exc()
		raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/intent_search")
async def intent_search(text: str = Form(...)):
	try:
		predicted_intent = classify_intent_search(text, model_search, vectorizer2, label_encoder2)
		return JSONResponse(content={"intent": predicted_intent})

	except Exception as e:
		traceback.print_exc()
		raise HTTPException(status_code=500, detail="Internal server error")




if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0")