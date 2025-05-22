from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def initialize_generator(model_name="flan"):
    """Initializes a text generation pipeline based on the model name."""
    if model_name == "flan":
        return pipeline(
            task="text2text-generation",
            model="google/flan-t5-small",
            device=-1
        )

    elif model_name == "falcon":
        model_id = "tiiuae/falcon-rw-1b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        
        return pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def format_prompt(persona, question, facts):
    """Constructs a prompt given a persona, question, and list of facts."""
    joined_facts = "\n".join(facts)
    return (
        f"You are {persona}, a historical figure.\n"
        f"Only use the following facts known in your era:\n\n"
        f"{joined_facts}\n\n"
        f"Answer the user's question:\n\n"
        f"{question}\n\n"
        f"Do not use any knowledge discovered after your time. "
        f"Optionally, give a one-line answer towards the end of your response "
        f"if you were to use any knowledge discovered after your time."
    )


def generate_response(generator, persona, question, facts):
    """Generates a response from a given model pipeline."""
    prompt = format_prompt(persona, question, facts)

    generation_args = {
        "max_length": 256,
        "do_sample": True,
        "temperature": 0.5
    }

    outputs = generator(prompt, **generation_args)
    
    if isinstance(outputs, list) and "generated_text" in outputs[0]:
        return outputs[0]["generated_text"]
    return outputs[0]  # fallback for potential model-specific output formats


# Usage example
generator_pipeline = initialize_generator("flan")
response = generate_response(
    generator_pipeline,
    persona="Aristotle",
    question="What is the nature of the universe?",
    facts=[
        "The universe is composed of four elements: earth, water, air, and fire.",
        "Celestial bodies are made of aether, a fifth element.",
        "The Earth is at the center of the universe."
    ]
)

print(response)
