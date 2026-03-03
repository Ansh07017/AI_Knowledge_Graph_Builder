import pandas as pd
import spacy

# Load SpaCy for efficiency in identifying standard entities
nlp = spacy.load("en_core_web_sm")

def get_deterministic_triplets(row):
    """
    Directly maps dataset columns to the defined structural triplet schema.
    """
    triplets = [
        {"subject": row['Customer Name'], "predicate": "PURCHASED", "object": row['Product Purchased']},
        {"subject": row['Customer Name'], "predicate": "RAISED", "object": f"Ticket_{row['Ticket ID']}"},
        {"subject": f"Ticket_{row['Ticket ID']}", "predicate": "REPORTS_ISSUE", "object": row['Ticket Subject']},
        {"subject": f"Ticket_{row['Ticket ID']}", "predicate": "HAS_SEVERITY", "object": row['Severity']},
        {"subject": f"Ticket_{row['Ticket ID']}", "predicate": "SUBMITTED_VIA", "object": row['Ticket Channel']},
        {"subject": f"Ticket_{row['Ticket ID']}", "predicate": "PRODUCED", "object": row['Resolution Status']}
    ]
    return triplets

def get_spacy_entities(text):
    """Extracts standard entities (dates, organizations) for efficiency."""
    if not nlp: return []
    doc = nlp(text)
    # Ensure 'ent' is defined inside the list comprehension loop
    return [{"subject": ent.text, "predicate": "MENTIONED_IN", "object": ent.label_} for ent in doc.ents]