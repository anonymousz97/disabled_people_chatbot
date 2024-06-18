# import streamlit as st
# from hugchat import hugchat
# from hugchat.login import Login
# from utils import get_result, detect_speech_end
# import json
# from api_call import *
# import numpy as np
# from ws import browser_func
# import simpleaudio as sa

# def play_wav(file_path):
# 	# Load the WAV file
# 	wave_obj = sa.WaveObject.from_wave_file(file_path)

# 	# Play the WAV file
# 	play_obj = wave_obj.play()

# 	# Wait for the playback to finish before exiting
# 	play_obj.wait_done()


# # Function to parse streaming data
# def parse_streaming_data(data):
# 	str_vl = data.decode('utf-8')
# 	text = ""
# 	if '"text":"' in str_vl:
# 		text = str_vl.split('"text":"')[1].split('"')[0]
# 		# Convert string literal to byte string
# 		byte_data = bytes(text, "latin1")

# 		# Decode byte string to Vietnamese string using UTF-8 encoding
# 		text = byte_data.decode('utf-8')
# 	return text

# # App title
# st.set_page_config(page_title="🤗💬 Viettel VTCC")

# if 'search_tool' not in st.session_state:
#     st.session_state.search_tool = None

# # Hugging Face Credentials
# with st.sidebar:
# 	st.title('🤗💬 Chatbot for disabled people')
# 	if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
# 		st.success('Login credentials already provided!', icon='✅')
# 		hf_email = st.secrets['EMAIL']
# 		hf_pass = st.secrets['PASS']
# 	else:
# 		hf_email = st.text_input('Enter E-mail:', type='password')
# 		hf_pass = st.text_input('Enter password:', type='password')
# 		if not (hf_email and hf_pass):
# 			st.warning('Please enter your credentials!', icon='⚠️')
# 		else:
# 			st.success('Proceed to entering your prompt message!', icon='👉')
# 	st.markdown('📖 Trợ lý ảo hỗ trợ người khuyết tật')
# 	if st.button("Reset"):
# 		st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]
# 		# st.experimental_rerun()
# 	if st.button("Ghi âm"):
# 		detect_speech_end()
# 		out = speech2text_vinai("recorded_audio.wav")
# 		st.write(out)
# 		st.session_state.messages.append({"role": "user", "content": str(out)})

		
 
	
# # Store LLM generated responses
# if "messages" not in st.session_state.keys():
# 	st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]

# # Display chat messages
# for message in st.session_state.messages:
# 	with st.chat_message(message["role"]):
# 		st.write(message["content"])
  

# # Function for generating LLM response
# # def generate_response(prompt_input, email, passwd):
# #     # Hugging Face Login
# #     sign = Login(email, passwd)
# #     cookies = sign.login()
# #     # Create ChatBot                        
# #     chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
# #     return chatbot.chat(prompt_input)

# # Browser

# def get_driver():
#     if st.session_state.search_tool is None:
#         search_tool = browser_func.Search()
#         st.session_state.search_tool = search_tool
#     return st.session_state.search_tool


# def generate_response(question, hf_email, hf_pass, stream=False):
# 	# retrieve 
# 	return get_result(question, stream)

# # User-provided prompt
# if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
# 	st.session_state.messages.append({"role": "user", "content": prompt})
# 	with st.chat_message("user"):
# 		st.write(prompt)


# # Generate a new response if last message is not from assistant
# if st.session_state.messages[-1]["role"] != "assistant":
# 	with st.chat_message("assistant"):
# 		# logic here
# 		intent = call_classify_intent_subintent(st.session_state.messages[-1]['content'])

# 		if intent['intent'] == 'chat' or intent['intent'] == 'search':
# 			response = st.write_stream(generate_response(st.session_state.messages[-1]['content'], hf_email, hf_pass, stream=True))
# 		elif intent['intent'] == 'create_image':
# 			response = st.write("Tạo ảnh...")
# 			from googletrans import Translator
# 			translator = Translator()
# 			r = translator.translate(st.session_state.messages[-1]['content'], dest='en').text
# 			a = create_image_api(r)
# 			a = np.array(eval(str(a)))
# 			st.image(a)
# 		else:
# 			print("Search tool: ", search_tool)
# 			search_tool = get_driver()

# 			# if "index" in intent.keys():
# 			# 	search_browser(intent['subintent'], intent['index'])
# 			# elif "query" in intent.keys():
# 			# 	search_browser(intent['subintent'], intent['query'])
# 			# else:
# 			# 	search_browser(intent['subintent'])

# 			if intent['subintent'] == 'search':
# 				search_tool.search(intent['query'])
# 			elif intent['subintent'] == 'scroll up':
# 				search_tool.scroll_up()
# 			elif intent['subintent'] == 'scroll down':
# 				search_tool.scroll_down()
# 			elif intent['subintent'] == 'back':
# 				search_tool.back()
# 			elif intent['subintent'] == 'choose link forward':
# 				search_tool.navigate(intent['index'])
# 			else:
# 				if st.session_state.search_tool:
# 					st.session_state.search_tool.quit()
# 					st.session_state.search_tool = None
# 			response = "Thực hiện thành công!"
# 		# # with st.spinner("Đang chuẩn bị câu trả lời..."):
# 		# #     response = generate_response(prompt, hf_email, hf_pass) 
# 		# #     st.markdown(response) 
# 	message = {"role": "assistant", "content": response}
# 	st.session_state.messages.append(message)
# 	text2speech(st.session_state.messages[-1]['content'])
# 	play_wav("audio_response.wav")


### NEW CODE
import streamlit as st
from hugchat import hugchat
from hugchat.login import Login
from utils import get_result, detect_speech_end
import json
from api_call import *
import numpy as np
from ws import browser_func
import simpleaudio as sa

def play_wav(file_path):
	# Load the WAV file
	wave_obj = sa.WaveObject.from_wave_file(file_path)
	# Play the WAV file
	play_obj = wave_obj.play()
	# Wait for the playback to finish before exiting
	play_obj.wait_done()

# Function to parse streaming data
def parse_streaming_data(data):
	str_vl = data.decode('utf-8')
	text = ""
	if '"text":"' in str_vl:
		text = str_vl.split('"text":"')[1].split('"')[0]
		# Convert string literal to byte string
		byte_data = bytes(text, "latin1")
		# Decode byte string to Vietnamese string using UTF-8 encoding
		text = byte_data.decode('utf-8')
	return text

# App title
st.set_page_config(page_title="🤗💬 Viettel VTCC")

if 'search_tool' not in st.session_state:
	st.session_state.search_tool = None

# Hugging Face Credentials
with st.sidebar:
	st.title('🤗💬 Chatbot for disabled people')
	if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
		st.success('Login credentials already provided!', icon='✅')
		hf_email = st.secrets['EMAIL']
		hf_pass = st.secrets['PASS']
	else:
		hf_email = st.text_input('Enter E-mail:', type='password')
		hf_pass = st.text_input('Enter password:', type='password')
		if not (hf_email and hf_pass):
			st.warning('Please enter your credentials!', icon='⚠️')
		else:
			st.success('Proceed to entering your prompt message!', icon='👉')
	st.markdown('📖 Trợ lý ảo hỗ trợ người khuyết tật')
	if st.button("Reset"):
		st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]
	if st.button("Ghi âm"):
		detect_speech_end()
		out = speech2text_vinai("recorded_audio.wav")
		st.write(out)
		st.session_state.messages.append({"role": "user", "content": str(out)})

# Store LLM generated responses
if "messages" not in st.session_state.keys():
	st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]

# Display chat messages
for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.write(message["content"])

# Function for generating LLM response
# def generate_response(prompt_input, email, passwd):
#     # Hugging Face Login
#     sign = Login(email, passwd)
#     cookies = sign.login()
#     # Create ChatBot                        
#     chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
#     return chatbot.chat(prompt_input)

# Browser
def get_driver():
	if st.session_state.search_tool is None:
		search_tool = browser_func.Search()
		st.session_state.search_tool = search_tool
	return st.session_state.search_tool

def generate_response(question, hf_email, hf_pass, stream=False):
	# retrieve 
	return get_result(question, stream)

# User-provided prompt
if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
	# Call the detect_speech_end function here
	detect_speech_end()
	out = speech2text_vinai("recorded_audio.wav")
	st.session_state.messages.append({"role": "user", "content": out})
	with st.chat_message("user"):
		st.write(out)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
	with st.chat_message("assistant"):
		# logic here
		intent = call_classify_intent_subintent(st.session_state.messages[-1]['content'])

		if intent['intent'] == 'chat':
			response = st.write_stream(generate_response(st.session_state.messages[-1]['content'], hf_email, hf_pass, stream=True))
		elif intent['intent'] == 'create_image':
			response = st.write("Tạo ảnh...")
			from googletrans import Translator
			translator = Translator()
			r = translator.translate(st.session_state.messages[-1]['content'], dest='en').text
			a = create_image_api(r)
			a = np.array(eval(str(a)))
			st.image(a)
		else:
			print("Search tool: ", search_tool)
			search_tool = get_driver()
			if intent['subintent'] == 'search':
				search_tool.search(intent['query'])
			elif intent['subintent'] == 'scroll up':
				search_tool.scroll_up()
			elif intent['subintent'] == 'scroll down':
				search_tool.scroll_down()
			elif intent['subintent'] == 'back':
				search_tool.back()
			elif intent['subintent'] == 'choose link forward':
				search_tool.navigate(intent['index'])
			else:
				if st.session_state.search_tool:
					st.session_state.search_tool.quit()
					st.session_state.search_tool = None
			response = "Thực hiện thành công!"
	message = {"role": "assistant", "content": response}
	st.session_state.messages.append(message)
	if intent['intent'] == 'chat':
		text2speech(st.session_state.messages[-1]['content'])
		play_wav("audio_response.wav")