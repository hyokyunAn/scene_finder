import openai
import ast
import json



def get_response(model_name, prompt, message):
    completion = openai.ChatCompletion.create(
    model=model_name,
    messages=[
        {"role": "user", "content": f"{prompt} {message}"}
    ]
    )

    return completion.choices[0].message["content"]




