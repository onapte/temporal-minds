from transformers import pipeline
import torch

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=0 if torch.cuda.is_available() else -1
)

def generate_answer(persona_name, query, retrieved_facts):
    context = "\n".join(retrieved_facts)
    prompt = f"""
        You are {persona_name}, a historical figure.
        Only use the following facts known in your era:

        {context}

        Answer the user's question:

        {query}

        Do not use any knowledge discovered after your time.
        """

    result = generator(prompt, max_length=256, do_sample=True, temperature=0.5)
    return result[0]['generated_text']
