import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

import os
from dotenv import load_dotenv
load_dotenv()

# 🔹 Configure Gemini API Key
genai.configure(api_key=os.getenv("API_KEY"))

# 🔹 Initialize LangChain Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("API_KEY"))

# 🔹 Memory to Remember Conversations
memory = ConversationBufferMemory()

# 🔹 Conversation Chain
chatbot = ConversationChain(llm=llm, memory=memory)

# 🔹 Streamlit UI
st.title("💬 AI Chatbot with LangChain & Gemini")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get AI response
    response = chatbot.invoke(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response["response"]})

    # Display response
    with st.chat_message("assistant"):
        st.write(response["response"])
