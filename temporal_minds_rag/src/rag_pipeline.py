from src.knowledge_loader import load_knowledge
from src.retriever import retrieve_facts
from src.generator import generate_answer

def rag_pipeline(persona_name, user_query):
    knowledge_base = load_knowledge(persona_name)
    retrieved_facts = retrieve_facts(knowledge_base, user_query)

    if not retrieved_facts:
        return f"Sorry, {persona_name} does not have enough historical knowledge to answer that."

    answer = generate_answer(persona_name, user_query, retrieved_facts)
    return answer