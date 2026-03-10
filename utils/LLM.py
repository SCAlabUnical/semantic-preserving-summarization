from openai import OpenAI

from constants import API_KEY


def ask_gpt(prompt, model_type="gpt-4o",temperature=0, top_p=0):
    client = OpenAI(api_key=API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_type,
        temperature=temperature,
        top_p=top_p,
    )
    response = chat_completion.choices[0].message.content

    return response