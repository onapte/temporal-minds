from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def load_generator(model_choice="flan"):
    device = -1

    if model_choice == "flan":
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device=device
        )

    elif model_choice == "falcon":
        model_id = "tiiuae/falcon-rw-1b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )

    else:
        raise NotImplementedError
    
generator = load_generator()   

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

    if generator.task == "text2text-generation":
        result = generator(prompt, max_length=256, do_sample=True, temperature=0.5)
        return result[0]['generated_text']

    elif generator.task == "text-generation":
        result = generator(prompt, max_length=256, do_sample=True, temperature=0.5)
        return result[0]['generated_text']

    else:
        raise NotImplementedError
