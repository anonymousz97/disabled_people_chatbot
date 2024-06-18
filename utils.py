# from transformers import AutoModelForCausalLM, AutoTokenizer
# from vllm import LLM, SamplingParams
# import torch
# from rank_bm25 import BM25Okapi
# import os
# import numpy as np

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')

# def gen_prompt(question, contexts):
#     '''
#     ### System:\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n\n\n### Instruction:\nYou are an AI assistant. Provide a detailed answer so user don’t need to search outside to understand the answer.\n\n### Input:\nDựa vào một số ngữ cảnh được cho dưới đây, trả lời câu hỏi ở cuối.\n\nContext [1]: Title: Thế vận hội Mùa hè 2020\nMặc dù được dời lại vào năm 2021, sự kiện này vẫn giữ tên \"Tokyo 2020\" cho mục đích tiếp thị và xây dựng thương hiệu. Đây là lần đầu tiên Thế vận hội đã bị hoãn lại và lên lịch lại, thay vì hủy bỏ. Tokyo đã được chọn là thành phố chủ nhà rong phiên họp IOC lần thứ 125 ở Buenos Aires, Argentina vào ngày 7 tháng 9 năm 2013. Thế vận hội 2020 sẽ đánh dấu lần thứ hai Nhật Bản tổ chức Thế vận hội Mùa hè, lần đầu tiên cũng ở Tokyo vào năm 1964, đánh dấu việc đây là thành phố đầu tiên ở châu Á tổ chức Thế vận hội Mùa hè hai lần và đồng thời đây là lần thứ tư Thế vận hội Mùa hè được tổ chức tại châu Á. Tính cả Thế vận hội Mùa đông vào năm 1972 (Sapporo) và năm 1998 (Nagano), đây là Thế vận hội thứ tư được tổ chức ở Nhật Bản. Thế vận hội 2020 cũng là Thế vận hội thứ hai trong ba kỳ Thế vận hội liên tiếp được tổ chức ở Đông Á, lần đầu tiên ở huyện Pyeongchang, Hàn Quốc vào năm 2018 à Thế vận hội tiếp theo ở Bắc Kinh, Trung Quốc vào năm 2022. Thế vận hội 2020 sẽ chứng kiến sự xuất hiện của các môn thi mới bao gồm bóng rổ 3x3, BMX tự do và xe đạp Madison, cũng như các sự kiện hỗn hợp tiếp theo. Theo các chính sách mới của IOC cho phép ban tổ chức chủ nhà thêm các môn thể thao mới vào chương trình Olympic để tăng cường các sự kiện cốt lõi vĩnh viễn, các Thế vận hội này sẽ chứng kiến karate, leo núi thể thao, lướt sóng và trượt ván có màn ra mắt tại Olympic, cũng như sự trở lại của bóng chày và bóng mềm lần đầu tiên kể từ năm 2008.\n\nQuestion: Nhật Bản đã tổ chức mấy kì Thế Vận hội mùa hè?\nHãy trả lời chi tiết và đầy đủ.\n\n### Response:\n""",
#     '''
#     system_prompt = "### System:\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n\n\n"
#     instruction = "### Instruction:\nYou are an AI assistant. Provide a detailed answer so user don’t need to search outside to understand the answer.\n\n"
#     input = "### Input:\nDựa vào một số ngữ cảnh được cho dưới đây, trả lời câu hỏi ở cuối.\n\n"
#     context = "\n".join(f"Context [{i+1}]: {x}" for i, x in enumerate(contexts))
#     question =  f"\n\nQuestion: {question}\nHãy trả lời chi tiết và đầy đủ.\n\n"
#     response =  "### Response:\n"
#     return system_prompt + instruction + input + context + question + response
	
# # sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
# # llm = LLM(model="../models/viettel_v3.2-awq", quantization='awq')
# llm = AutoModelForCausalLM.from_pretrained("minhbui/viettel_v3.2", torch_dtype=torch.bfloat16, load_in_8bit=True).cuda()
# tokenizer = AutoTokenizer.from_pretrained("minhbui/viettel_v3.2")

# def get_topk_embedding(question, context, model, topk=5):
#     ctx = [tokenize(sentence) for sentence in context]
#     embeddings = model.encode(ctx)
#     embed_question = model.encode([tokenize(question)])[0]
#     embed_question /= np.linalg.norm(embed_question)
#     embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

#     # Compute cosine similarities
#     cosine_similarities = np.dot(embeddings, embed_question)

#     # Get the indices of the top 5 most similar documents
#     top_k_indices = np.argsort(cosine_similarities)[::-1][:topk].tolist()
#     scores = cosine_similarities.tolist()
#     scores.sort(reverse=True)
#     lst_res = []
#     for i in top_k_indices:
#         lst_res.append(context[i])
#     return lst_res, top_k_indices, scores[:topk]
	

# # def gen_answer(prompts, sampling_params=sampling_params):
# #     outputs = llm.generate(prompts, sampling_params)
# #     return outputs[0].outputs[0].text

# from transformers import GenerationConfig
# def gen_answer(prompt, max_new_tokens=768):
#     input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(llm.device)
#     llm.eval()
#     with torch.no_grad():
#         generation_config = GenerationConfig( repetition_penalty=1.1,max_new_tokens=max_new_tokens,temperature=0.2,top_p=0.95,top_k=40,
# 	   # bos_token_id=tokenizer.bos_token_id,
# 	         # eos_token_id=tokenizer.eos_token_id,
# 	        # eos_token_id=0, # for open-end generation.
# 	        pad_token_id=tokenizer.pad_token_id,
# 	        do_sample=True,
# 	        use_cache=True,
# 	        return_dict_in_generate=True,
# 	        output_attentions=False,
# 	        output_hidden_states=False,
# 	        output_scores=False,
# 	        )
#         generated = llm.generate(inputs=input_ids,generation_config=generation_config)
#         gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
#         output = tokenizer.batch_decode(gen_tokens)[0]
#         output = output.split(tokenizer.eos_token)[0]
#         return output.strip()

# def split_text_into_chunks(text, chunk_size=100, window_size=50):
#     words = text.split()
#     num_words = len(words)
#     chunks = []
#     start_idx = 0

#     while True:
#         end_idx = start_idx + chunk_size
#         chunk = " ".join(words[start_idx:end_idx])
#         chunks.append(chunk)
#         if end_idx >= num_words:
#             break
#         start_idx += window_size

#     return chunks

# corpus = []
# title_corpus = []
# for fil in os.listdir("../../db/"):
#     if not fil.endswith("txt"):
#         print(fil)
#         continue
#     tmp_file = os.path.join('../../db',fil)
#     with open(tmp_file,'r') as f:
#         text = f.read()
#         split_chunks = split_text_into_chunks(text,150,150)
#         corpus += split_chunks
#         title_corpus += [fil] * len(split_chunks) 


# tokenized_corpus = [doc.split(" ") for doc in corpus]

# bm25 = BM25Okapi(tokenized_corpus)

# import pickle
# with open('../../db/corpus_embedding_w150.pkl', 'rb') as f:
#     corpus_embedding = pickle.load(f)

# def retrieve(question, topk=2, semantic=True):
#     if semantic:
#         embed_question = model.encode([question])[0]
#         #print(embed_question.shape, type(embed_question))
#         embed_question /= np.linalg.norm(embed_question)
#         # Compute cosine similarities
#         cosine_similarities = np.dot(corpus_embedding, embed_question)
#         # Get the indices of the top 5 most similar documents
#         top_k_indices = np.argsort(cosine_similarities)[::-1][:topk].tolist()
#         result = []
#         result_title = []
#         for i in top_k_indices:
#             result.append(corpus[i])
#             result_title.append(title_corpus[i])
#         return result, result_title
		
#     tokenized_query = question.split(" ")
#     return bm25.get_top_n(tokenized_query, corpus, n=topk), []

# def add_citation(answer, lst_citations):
#     adding_citations_string = []
#     lst_citations = list(dict.fromkeys(lst_citations))
#     if len(lst_citations) == 1:
#         adding_citations_string.append(f'Source : {lst_citations[0]}')
#     else:
#         for idx in range(len(lst_citations)):
#             if f"[{idx+1}]" in answer:
#                 adding_citations_string.append(f'Source : {lst_citations[idx]}')
#     answer += "\n\n"
#     answer += "\n".join(str(x) for x in adding_citations_string)
#     return answer
			
			

# def get_result(question):
#     contexts, retrieved_title = retrieve(question)
#     print(contexts, retrieved_title)
#     prompt = gen_prompt(question, contexts)


#     # answer = gen_answer(prompt, sampling_params)
#     answer = gen_answer(prompt)
#     answer = add_citation(answer, retrieved_title)
#     return answer

# # print(retrieve("Viettel được thành lập năm nào?"))


import requests
import json
import time

temp_current = ""

def preprocess(chunk):
	global temp_current
	if '\n\n'.encode('utf-8') not in chunk:
		if temp_current == "":
			temp_current = chunk
		else:
			temp_current += chunk
		return ""
	else:
		tmp_answer = ""
		print("chunk", chunk)
		idx = chunk.index('\n\n'.encode('utf-8'))
		temp_current += chunk[:idx]
		res = temp_current.decode('utf-8')
		while True:
			if len(res) < 1:
				break
			if res[0] != '{':
				res = res[1:]
			else:
				break
		print("res", res)
		res = res.split('\n\n')
		for i in res:
			if len(i) < 1:
				continue
			while True:
				if len(i) < 1:
					break
				if i[0] != '{':
					i = i[1:]
				else:
					break
			if len(i) < 1:
				continue
			tmp = json.loads(i)['choices'][0]['text']
			tmp_answer += tmp
		temp_current = chunk[idx:]
		return tmp_answer


url = "https://api.together.xyz/v1/chat/completions"

def process_sample(query, stream=False):
	print("query", query)
	payload = {
		"model": "meta-llama/Llama-3-70b-chat-hf",
		"temperature": 1,
		"frequency_penalty": 0,
		"presence_penalty": 0,
		"stream_tokens": True,
		"messages": [
			{
				"role": "user",
				"content": "Bạn là trợ lý ảo dành cho người Việt do minhbc4 tạo ra, hãy giúp người dùng trả lời câu hỏi 1 cách chính xác và cẩn thận. Mọi trả lời phải được trả lời bằng tiếng Việt và không lẫn tiếng nước ngoài vào.\n"+query
			}
		],
		"stream": stream,
		"stop": ["<|eot_id|>", "<|end_of_text|>"],
		"max_tokens": 1024
	}
	headers = {
		"accept": "application/json",
		"content-type": "application/json",
		"Authorization": "Bearer ee13c239c422e600d2f4bbf230f1dd6c4bd1bdcb7cffbc82200874e71783080c"
	}

	response = requests.post(url, json=payload, headers=headers)
	for chunk in response:
		time.sleep(0.05)
		res = preprocess(chunk)
		if res != "":
			yield res
	temp_current = ""

def get_result(query, stream):
	return process_sample(query, stream=stream)


import speech_recognition as sr
import wave

def save_audio_to_wav(audio_data, filename):
	# Save the audio data to a WAV file
	with wave.open(filename, 'wb') as f:
		f.setnchannels(1)  # Mono
		f.setsampwidth(audio_data.sample_width)
		f.setframerate(audio_data.sample_rate)
		f.writeframes(audio_data.get_wav_data())

def detect_speech_end():
	recognizer = sr.Recognizer()
	microphone = sr.Microphone()

	with microphone as source:
		print("Listening for speech...")
		recognizer.adjust_for_ambient_noise(source)
		
		# Listen for the user's speech until a pause is detected
		audio_data = recognizer.listen(source, timeout=5)
		print("Recording complete")

		# Save the audio data to a WAV file
		save_audio_to_wav(audio_data, "recorded_audio.wav")
		

def process_sample_no_stream(query):
	# print("query", query)
	payload = {
		"model": "meta-llama/Llama-3-70b-chat-hf",
		"temperature": 1,
		"frequency_penalty": 0,
		"presence_penalty": 0,
		"messages": [
			{
				"role": "user",
				"content": "Bạn là trợ lý ảo dành cho người Việt do minhbc4 tạo ra, hãy giúp người dùng trả lời câu hỏi 1 cách chính xác và cẩn thận. Mọi trả lời phải được trả lời bằng tiếng Việt và không lẫn tiếng nước ngoài vào.\n"+query
			}
		],
		"stop": ["<|eot_id|>", "<|end_of_text|>"],
		"max_tokens": 1024
	}
	headers = {
		"accept": "application/json",
		"content-type": "application/json",
		"Authorization": "Bearer ee13c239c422e600d2f4bbf230f1dd6c4bd1bdcb7cffbc82200874e71783080c"
	}

	response = requests.post(url, json=payload, headers=headers)
	# print(response.text)
	return json.loads(response.text)["choices"][0]["message"]['content']