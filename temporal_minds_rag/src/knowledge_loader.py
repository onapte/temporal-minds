import os
import json

def load_knowledge(persona_name):
    filepath = f'data/{persona_name}_knowledge.json'

    if not os.path.exists(filepath):
        raise FileNotFoundError("Knowledge file not found")
    
    with open(filepath, 'r') as f:
        return json.load(f)