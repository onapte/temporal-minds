from src.knowledge_loader import load_knowledge
from src.retriever import Retriever
from src.generator import generate_answer

def rag_pipeline(persona_name, user_query):
    knowledge_base = load_knowledge(persona_name)
    
    retriever = Retriever(embed_fields=["text"])
    retriever.build_index(knowledge_base)

    retrieved_facts = [doc["text"] for doc in retriever.query(user_query, top_k=3)]
    if not retrieved_facts:
        return f"Sorry, {persona_name} does not have enough historical knowledge to answer that!"

    answer = generate_answer(persona_name, user_query, retrieved_facts)
    return answer