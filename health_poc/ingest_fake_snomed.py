import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

URI = "bolt://localhost:7687" 
AUTH = ("", "")

print("Loading AI Model (all-mpnet-base-v2)...")
model = SentenceTransformer('all-mpnet-base-v2')
print("Model Loaded.")

def get_embedding(text):
    return model.encode(text).tolist()

def ingest():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    with driver.session() as session:
        print("Clearing old data...")
        session.run("MATCH (n) DETACH DELETE n")
        
        print("Creating Vector Index (768 dimensions)...")
        try:
            session.run("DROP VECTOR INDEX snomed_description_index")
        except:
            pass
        
        # Create index on :Description(embedding)
        session.run("""
            CREATE VECTOR INDEX snomed_description_index ON :Description(embedding) 
            WITH CONFIG {"dimension": 768, "metric": "cos", "capacity": 10000}
        """)

        print("Injecting Clinical Concepts...")
        
        queries = [
            # Symptoms
            "CREATE (c:Concept {id: '29857009'}) CREATE (d:Description {term: 'Chest pain', type: 'Synonym'}) CREATE (c)-[:HAS_DESCRIPTION]->(d)",
            "CREATE (c:Concept {id: '422587007'}) CREATE (d:Description {term: 'Nausea', type: 'Synonym'}) CREATE (c)-[:HAS_DESCRIPTION]->(d)",
            "CREATE (c:Concept {id: '162059005'}) CREATE (d:Description {term: 'Upset stomach', type: 'Synonym'}) CREATE (c)-[:HAS_DESCRIPTION]->(d)",
            "CREATE (c:Concept {id: '404640003'}) CREATE (d:Description {term: 'Dizziness', type: 'Synonym'}) CREATE (c)-[:HAS_DESCRIPTION]->(d)",
            
            # Conditions
            "CREATE (c:Concept {id: '22298006', active: true}) CREATE (d:Description {term: 'Myocardial infarction', type: 'FSN'}) CREATE (c)-[:HAS_DESCRIPTION]->(d)",
            "CREATE (c:Concept {id: '235595009', active: true}) CREATE (d:Description {term: 'Gastroesophageal reflux disease', type: 'FSN'}) CREATE (c)-[:HAS_DESCRIPTION]->(d)",
            "CREATE (c:Concept {id: '73410007', active: true}) CREATE (d:Description {term: 'Panic attack', type: 'FSN'}) CREATE (c)-[:HAS_DESCRIPTION]->(d)",
            
            # Relationships (Logic)
            # MI causes Chest Pain + Nausea + Dizziness
            "MATCH (c:Concept {id: '22298006'}), (s:Concept {id: '29857009'}) CREATE (c)-[:ASSOCIATED_WITH]->(s)",
            "MATCH (c:Concept {id: '22298006'}), (s:Concept {id: '422587007'}) CREATE (c)-[:ASSOCIATED_WITH]->(s)",
            "MATCH (c:Concept {id: '22298006'}), (s:Concept {id: '404640003'}) CREATE (c)-[:ASSOCIATED_WITH]->(s)",
            
            # GERD causes Chest Pain + Upset Stomach (But usually NOT dizziness)
            "MATCH (c:Concept {id: '235595009'}), (s:Concept {id: '29857009'}) CREATE (c)-[:ASSOCIATED_WITH]->(s)",
            "MATCH (c:Concept {id: '235595009'}), (s:Concept {id: '162059005'}) CREATE (c)-[:ASSOCIATED_WITH]->(s)",

            # 2. ORTHOPEDIC (The "Knee" Cluster)
            "CREATE (c:Concept {id: '239516002'}) CREATE (d:Description {term: 'Knee pain', type: 'Synonym'}) CREATE (c)-[:HAS_DESCRIPTION]->(d)",
            "CREATE (c:Concept {id: '125667009'}) CREATE (d:Description {term: 'Bruising', type: 'Synonym'}) CREATE (c)-[:HAS_DESCRIPTION]->(d)",
            
            # Conditions
            "CREATE (c:Concept {id: '300860001', active: true}) CREATE (d:Description {term: 'Sprain of knee', type: 'FSN'}) CREATE (c)-[:HAS_DESCRIPTION]->(d)",
            "CREATE (c:Concept {id: '239720002', active: true}) CREATE (d:Description {term: 'Osteoarthritis of knee', type: 'FSN'}) CREATE (c)-[:HAS_DESCRIPTION]->(d)",
            
            # Relationships
            "MATCH (c:Concept {id: '300860001'}), (s:Concept {id: '239516002'}) CREATE (c)-[:ASSOCIATED_WITH]->(s)",
            "MATCH (c:Concept {id: '300860001'}), (s:Concept {id: '125667009'}) CREATE (c)-[:ASSOCIATED_WITH]->(s)",
            "MATCH (c:Concept {id: '239720002'}), (s:Concept {id: '239516002'}) CREATE (c)-[:ASSOCIATED_WITH]->(s)",
        ]
        
        for q in queries:
            session.run(q)
            
        print("Generating & Storing Embeddings...")

        result = session.run("MATCH (d:Description) RETURN id(d) as id, d.term as term")
        nodes = list(result)
        
        for node in nodes:
            vector = get_embedding(node["term"])
            session.run(
                "MATCH (d:Description) WHERE id(d) = $id SET d.embedding = $vector",
                id=node["id"], vector=vector
            )
            
    print(f"Success! Ingested {len(nodes)} nodes with 768-dim embeddings.")

if __name__ == "__main__":
    ingest()