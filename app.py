import os
import pandas as pd
from flask import Flask, jsonify, render_template
from dotenv import load_dotenv
from neo4j import GraphDatabase

from triplet_definition import get_deterministic_triplets
from llm_engine import extract_semantic_triplets

load_dotenv()
app = Flask(__name__)

# Neo4j Driver Setup (Ensure these match your .env)
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(user, password))

# Load a small sample to test the live visualization effectively
try:
    df = pd.read_csv("cleaned_tickets.xlsx - Sheet1.csv").head(5)
except Exception as e:
    print(f"Error loading CSV. Make sure the filename is exact. {e}")
    df = pd.DataFrame() # Fallback empty dataframe

def ingest_to_neo4j(triplets):
    """Pushes triplets to the Neo4j Database."""
    try:
        with driver.session() as session:
            for t in triplets:
                query = (
                    "MERGE (s:Entity {name: $sub}) "
                    "MERGE (o:Entity {name: $obj}) "
                    "MERGE (s)-[:`" + str(t['predicate']) + "`]->(o)"
                )
                session.run(query, sub=str(t['subject']), obj=str(t['object']))
    except Exception as e:
        print(f"Neo4j Ingestion Error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_graph')
def process_graph():
    results = []
    
    if df.empty:
        return jsonify({"error": "No data found to process."}), 400

    for _, row in df.iterrows():
        # 1. Structural Logic
        structural = get_deterministic_triplets(row)
        
        # 2. Semantic Logic (LLM)
        semantic = extract_semantic_triplets(str(row.get('Ticket Description', '')))
        
        # 3. Action: Ingest to Neo4j
        all_triplets = structural + semantic
        ingest_to_neo4j(all_triplets)
        
        # Package data for the frontend dashboard
        results.append({
            "ticket_id": str(row.get('Ticket ID', 'Unknown')),
            "raw_text": str(row.get('Ticket Description', 'No description provided.')),
            "structural_triplets": structural,
            "semantic_triplets": semantic
        })
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000, debug=True)