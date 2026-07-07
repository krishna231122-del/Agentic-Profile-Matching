"""
main.py
Entry point for the AI Recruiting Matching Agent.
"""

import os
from Agents import matching_agent
from Agents.matching_agent import MatchingAgent, vector_store

def main():
    # 1. Check actual document count, not folder existence
    #    (Chroma creates the folder on import even with 0 docs, so
    #    os.path.exists() is always True and useless as a check)
    doc_count = vector_store._collection.count()
    print(f"Current vector DB doc count: {doc_count}")

    if doc_count == 0:
        print("Indexing resumes into vector database...")
        print(matching_agent.index_resumes())
        print("Doc count after indexing:", vector_store._collection.count())
    else:
        print("Vector DB already has documents, skipping indexing.")

    # 2. Initialize the agent
    agent = matching_agent.MatchingAgent()

    # 3. Provide job description and run initial screening (Round 1)
    jd = """
    Senior AI Engineer / ML Engineer / Deep Learning Engineer / MLOps Engineer
    Must have: C++, Python, Machine Learning, Deep Learning, MLOps, 3+ years experience, FastAPI integration
    Nice to have: MTech, GraphQL, Next.js, testing frameworks (Jest/Cypress)
    """
    result = agent.start_screening(jd, round_number=1)
    print("\n--- Initial Screening Report ---")
    print(result["messages"][-1].content)

    # 4. Interactive loop for natural language queries
    print("\nAgent ready. Type your request (or 'exit' to quit):")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        if user_input.lower() in ("next round", "advance round"):
            result = agent.run_next_round()
        else:
            result = agent.send_feedback(user_input)

        print(f"\nAgent: {result['messages'][-1].content}")

    # 5. Print final report at the end
    print("\n--- Final Report ---")
    print(agent.get_final_report())

if __name__ == "__main__":
    main()