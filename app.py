import streamlit as st

st.title("Personal Info Chatbot")
st.write("Welcome to the Personal Info Chatbot! This chatbot is designed to retrieve information about Sim Yew Chong. You can ask questions about his background, education, and more. Feel free to ask anything you'd like to know!")

## Tools portion
import os
from typing import List, Dict, Union
import openai
from openai import AzureOpenAI, AsyncAzureOpenAI, AsyncOpenAI, OpenAI
import json
import re
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")



