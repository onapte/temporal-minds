from src.rag_pipeline import rag_pipeline

if __name__ == "__main__":
    persona = "aristotle"
    query = "What is the soul?"

    print("\n=== Temporal Minds - Historical Response ===\n")
    result = rag_pipeline(persona, query)
    print(result)
