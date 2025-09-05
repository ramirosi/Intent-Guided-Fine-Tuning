import numpy as np
import requests
import json 

def call_GPT(prompt, temperature=0.7, check_time = False, return_token_count = False):

  # FILL IN WITH YOUR OWN API OR OTHER LLM
  
  url = ''

  headers = ''

  data = {
      "messages": [{"role": "system", "content": f"{prompt}"}],
      "max_tokens": 3000, # Increased otherwise truncated output
      "temperature": temperature, # Variability choice
      "frequency_penalty": 0,
      "presence_penalty": 0,
      "top_p": 0.95,
      "stop": None
      }
  
  response = requests.post(url, headers=headers, data=json.dumps(data))

  if check_time:
    return json.loads(response.text)
  elif return_token_count:
    return json.loads(response.text)["choices"][0]["message"]["content"], json.loads(response.text)['usage']['total_tokens']
  else:
    return json.loads(response.text)["choices"][0]["message"]["content"] # Return just the message

def call_ADA_003(msg):
    
    # FILL IN WITH YOUR OWN API OR OTHER EMBEDDER MODEL
    url = ""

    headers = ""

    data = {
        "input": [f"{msg}"]
        }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    return json.loads(response.text)["data"][0]["embedding"]

