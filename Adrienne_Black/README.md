Assignment 1_Day 2
!pip install huggingface_hub

!pip install -U gradio

!pip install transformers accelerate sentencepiece -q

from huggingface_hub import whoami

from huggingface_hub import login

import gradio as gr

from transformers import pipeline

from google.colab import userdata

hf_token=userdata.get('HF_TOKEN')

login=hf_token

try:
  user_info=whoami(token=hf_token)
  print(f"Logged in as: {user_info["name"]}")
except:
  print("Login failed. Token is required")

summarizer = pipeline(task="text-generation", model="facebook/bart-large-cnn")

def summarize(inputTextForSummarization):
  name=summarize(inputTextForSummarization, max_length=100, min_length=10)
  return name[0]['summary_text']

demo=gr.Interface(fn=summarize, inputs="text",outputs="text")

demo.launch(share=True)
