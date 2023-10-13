import streamlit as st
from streamlit_chat import message

import cohere
global arr

def clear_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Say something to get started!"}]

st.set_page_config(page_title="Search Optimizer", page_icon="👀")

st.title("Search Optimizer")

st.markdown(
    "This is demo showcase of Cohere Rerank (Beta) model. It allows to optimize search algorithms and improve search results."
)

with st.sidebar:
    arr = []
    cohere_api_key = st.text_input('Cohere API Key', key='cohere_api_key')
    "Don't have API Key? [Get for free](https://dashboard.cohere.com/api-keys)"
    "[View the source code](https://github.com/abdibrokhim/Cohere-Rerank-Tutorial)"

    file = st.file_uploader(label="Upload file", type=["txt",])
    if file:
        try:
            filename = "file.txt"
            with open(filename, "wb") as f:
                f.write(file.getbuffer())

            with open(filename, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    print(f"line: {line}")
                    arr.append(line)
        except FileNotFoundError:
            print(f"File not found: {file}")
        except Exception as e:
            print(f"An error occurred: {e}")



st.title("PaLM Tutorial")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Say something to get started!"}]


with st.form("chat_input", clear_on_submit=True):
    a, b = st.columns([4, 1])

    user_prompt = a.text_input(
        label="Your message:",
        placeholder="Type something...",
        label_visibility="collapsed",
    )

    b.form_submit_button("Send", use_container_width=True)


for msg in st.session_state.messages:
    message(msg["content"], is_user=msg["role"] == "user")


if user_prompt and not cohere_api_key:
    st.info("Please add your PaLM API key to continue.")


# uncomment "docs" for quick test
# docs = [
#     "Carson City is the capital city of the American state of Nevada. At the 2010 United States Census, Carson City had a population of 55,274.",
#     "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
#     "Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
#     "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district. The President of the USA and many major national government offices are in the territory. This makes it the political center of the United States of America.",
#     "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states. The federal government (including the United States military) also uses capital punishment."
# ]

if user_prompt:

    print('user_prompt: ', user_prompt)

    co = cohere.Client(cohere_api_key) # Paste your own API key here

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    message(user_prompt, is_user=True)

    response = co.rerank(
        query=user_prompt, 
        documents=arr, 
        top_n=3,
        model='rerank-english-v2.0'
    )
    print(f"response: {response}")

    info = []

    for idx, r in enumerate(response):
        info.append(f"""
Document Rank: {idx + 1}
Document Index: {r.index}
Document: {r.document['text']}
Relevance Score: {r.relevance_score:.2f}
\n
""")

    msg = {"role": "assistant", "content": info}

    print('st.session_state.messages: ', st.session_state.messages)

    st.session_state.messages.append(msg)

    print('msg.content: ', msg["content"])

    for msgs in msg["content"]:
        message(msgs, is_user=False, key=msgs)


if len(st.session_state.messages) > 1:
    st.button('Clear Chat', on_click=clear_chat)
    