from src.rag_pipeline import rag_pipeline

if __name__ == "__main__":
    persona = "aristotle"
    query = "What did Aristotle propose?"

    print("\nResponse\n")
    result = rag_pipeline(persona, query)
    print(result)
