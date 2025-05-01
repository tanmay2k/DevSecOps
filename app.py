import os
from openai import OpenAI


# Retrieve the environment variables
api_key = os.getenv("LLM_API_KEY")


client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

completion = client.chat.completions.create(
  extra_body={},
  model="meta-llama/llama-3.3-70b-instruct",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices[0].message.content)
