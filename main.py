# 챗봇

import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai
import google.ai.generativelanguage as glm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import json
from PIL import Image
import matplotlib.pyplot as plt

from one_pic_infer import one_pic_inference

model_name = 'SBERT' # choose between 'gemini-pro' and 'SBERT'

safety_settings={
  'harassment':'BLOCK_MEDIUM_AND_ABOVE',
  'hate':'BLOCK_MEDIUM_AND_ABOVE',
  'sex':'BLOCK_MEDIUM_AND_ABOVE',
  'danger':'BLOCK_MEDIUM_AND_ABOVE'
}

cfg = genai.GenerationConfig(
    candidate_count = 1,
    stop_sequences = None,
    max_output_tokens = None,
    temperature = 0.,
    top_k = 40,
    top_p = 1,
)

@st.cache_resource
def get_model():
  if model_name == 'gemini-pro':
    model = genai.GenerativeModel(
      model_name= model_name,
      generation_config=cfg,
      safety_settings=safety_settings,
    )
  
  elif model_name == 'SBERT':
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

  return model

@st.cache_data
def read_data():
  if model_name == 'gemini-pro':
    df = pd.read_csv('./data/gemini_data.csv')
  elif model_name == 'SBERT':
    df = pd.read_csv('./data/sbert_datat.csv')
  df['Embeddings'] = df['Embeddings'].apply(json.loads)
  return df

def stream_display(response, placeholder):
  text=''
  for chunk in response:
    if parts:=chunk.parts:
      if parts_text:=parts[0].text:
        text += parts_text
        placeholder.write(text + "▌")
  return text

def init_messages() -> None:
  st.session_state.messages = []

def set_generate(state=True):
  st.session_state.generate = state

def find_best_passage(query, dataframe):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  query_embedding = genai.embed_content(model='models/embedding-001',
                                        content=query,
                                        task_type="retrieval_query")
  
  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])

  idx = np.argmax(dot_products)
  return dataframe.iloc[idx]['consulting'] # Return text from index with max value

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ") # preprocess
  prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond being comprehensive\
  strike a friendly and converstional tone. \
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt

model = get_model()
df = read_data()

# Google API key
if "api_key" not in st.session_state:
  try:
    st.session_state.api_key = st.secrets["GOOGLE_API_KEY"]
  except:
    st.session_state.api_key = ""
    st.write("Your Google API Key is not provided in `.streamlit/api_key.toml`")


st.title("💬 심리상담 Chatbot")

if "messages" not in st.session_state:
  init_messages()
  st.session_state.messages.append(
    glm.Content(role='model', parts=[glm.Part(text='무슨 고민이 있으세요?')])
  )
  set_generate(False)

# sidebar
with st.sidebar:

  # # model
  # st.header("MODEL")
  # model_name = st.selectbox("model_name", ['gemini-pro', 'SBERT'])

  # clear
  st.header("Clear")
  st.button("Clear", on_click=init_messages, use_container_width=True)

    # Image
  st.header("Image")
  image_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
  if image_file is not None:
    image = Image.open(image_file)
    #st.image(image, use_column_width=True)
    #st.sidebar.image(image, use_column_width=True, caption="Uploaded! :)")

  ##### image inference #####
  fig, ax, emotion= one_pic_inference(image)

  # Display the image
  st.pyplot(fig)

  # 결과 출력
  st.write(f"당신의 지금 감정은 {emotion} 입니다.")

# Display messages in history
for content in st.session_state.messages:
  if text := content.parts[0].text:
    with st.chat_message('human' if content.role == 'user' else 'ai'):
      st.write(text)

# Chat input
if prompt := st.chat_input("저에게 말 해 보세요."):
  set_generate(True)
  # Append to history
  st.session_state.messages.append(
    glm.Content(role='user', parts=[glm.Part(text=prompt)])
  )
  # Display input message
  with st.chat_message('human'):
    st.write(prompt)

# AI generate
if st.session_state.generate:
  set_generate(False)
  # Generate

  if model_name == 'gemini-pro':
    passage = find_best_passage(prompt, df)
    prompted = make_prompt(prompt, passage)
    response = model.generate_content(prompted, stream=True)
  
  elif model_name == 'SBERT':
    embedded = model.encode(prompt)
    df['distance'] = df['Embeddings'].map(lambda x: cosine_similarity([embedded], [x]).squeeze())
    text = df.loc[df['distance'].idxmax()]['consulting']

  
  # Stream display
  with st.chat_message("ai"):
    placeholder = st.empty()

  if model_name == 'gemini-pro':
    text = stream_display(response, placeholder)
    # Append to history
    st.session_state.messages.append(response.candidates[0].content)

  elif model_name == 'SBERT':
    st.session_state.messages.append(
    glm.Content(role='ai', parts=[glm.Part(text=text)])
  )

  placeholder.write(text)