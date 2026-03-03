import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

# Initialize Llama 3.1 using the .env key
llm = ChatGroq(
    model_name="llama3-8b-8192", 
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0 
)

PROMPT_TEMPLATE = """
System: You are a Knowledge Graph Specialist. Extract technical triplets from the description.
Description: "{description}"

Rules:
1. Extract [Product] —[:EXPERIENCING]—> [Specific Error/Bug]
2. Extract [Product] —[:COMPONENT_INVOLVED]—> [Hardware/Software Part]
3. Extract [Resolution] —[:REQUIRED_ACTION]—> [Troubleshooting Step]

Format: Subject | Predicate | Object (One per line)
"""

def extract_semantic_triplets(description):
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    
    try:
        response = chain.invoke({"description": description})
        raw_lines = response.content.strip().split("\n")
        
        semantic_triplets = []
        for line in raw_lines:
            if "|" in line:
                parts = line.split("|")
                if len(parts) == 3:
                    semantic_triplets.append({
                        "subject": parts[0].strip(),
                        "predicate": parts[1].strip(),
                        "object": parts[2].strip()
                    })
        return semantic_triplets
    except Exception as e:
        print(f"Error in LLM Extraction: {e}")
        return []