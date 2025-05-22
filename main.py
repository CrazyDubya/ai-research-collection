from markdown import markdown
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Google API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Google Generate
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("The opposite of hot is")
print(response.text)

# Google Image
img = Image.open('test.jpeg')
prompt = """This image contains a sketch of a potential product along with some notes.
Given the product sketch, describe the product as thoroughly as possible based on what you
see in the image, making sure to note all of the product features. Return output in json format:
{description: description, features: [feature1, feature2, feature3, etc]}"""

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content([prompt, img])
print(response.text)

#Google Video (Can also upload img and sound this way)

video_file_name = "BigBuckBunny_320x180.mp4"

print(f"Uploading file...")
video_file = genai.upload_file(path=video_file_name)
print(f"Completed upload: {video_file.uri}")

import time

while video_file.state.name == "PROCESSING":
    print('Waiting for video to be processed.')
    time.sleep(10)
    video_file = genai.get_file(video_file.name)

if video_file.state.name == "FAILED":
    raise ValueError(video_file.state.name)
print(f'Video processing complete: ' + video_file.uri)
# Create the prompt.
prompt = "Describe this video."

# Set the model to Gemini 1.5 Flash.
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# Make the LLM request.
print("Making LLM inference request...")
response = model.generate_content([prompt, video_file],
                                  request_options={"timeout": 600})
print(response.text)


# Google Function Calling
def add(a: float, b: float):
    """returns a + b."""
    return a + b

def subtract(a: float, b: float):
    """returns a - b."""
    return a - b

def multiply(a: float, b: float):
    """returns a * b."""
    return a * b

def divide(a: float, b: float):
    """returns a / b."""
    return a / b

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", tools=[add, subtract, multiply, divide]
)

chat = model.start_chat(enable_automatic_function_calling=True)
response = chat.send_message(
    "I have 57 cats, each owns 44 mittens, how many mittens is that in total?"
)
print(response.text)

# Google Grounding
model = genai.GenerativeModel('models/gemini-1.5-pro-002', tools="google_search_retrieval")
chat = model.start_chat()

result = chat.send_message('what is the area of spain?')
print(markdown(result.text))

html_content = result.candidates[0].grounding_metadata.search_entry_point.rendered_content
print(html_content)

result = chat.send_message('what is the google stock price?')
print(markdown(result.text))

html_content = result.candidates[0].grounding_metadata.search_entry_point.rendered_content
print(html_content)

