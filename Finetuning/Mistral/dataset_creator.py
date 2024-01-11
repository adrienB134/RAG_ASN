from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

for file in os.listdir(f"Ingestion/ASN/lettres_de_suivi/txt"):
    with open(f"Ingestion/ASN/lettres_de_suivi/txt/{file}") as f:
        raw_text = f.read()

    pairs = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": """You are an expert AI assisting us in creating a high quality, diverse synthetic dataset to train Information Retrieval models. 
                    Your role is to analyse the document given to you and provide us with 20 high quality questions and answers from the document.
                    Your answer should answer in the following format:
                    "[INST] here should be the question [/INST] here the Answer "
                    The document is in french so answer in french.""",
            },
            {"role": "user", "content": raw_text},
        ],
    )
    with open("dataset.txt", "a") as f:
        f.write(pairs.choices[0].message.content)

    time.sleep(5)
