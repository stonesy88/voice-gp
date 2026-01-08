import os
import csv
import sys
import logging
import gc
import time
import torch
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "snomed")

CONCEPT_FILE = os.path.join(BASE_PATH, "sct2_Concept_MONOSnapshot_GB_20251217.txt")
DESC_FILE = os.path.join(BASE_PATH, "sct2_Description_MONOSnapshot-en_GB_20251217.txt")
REL_FILE = os.path.join(BASE_PATH, "sct2_Relationship_MONOSnapshot_GB_20251217.txt")

DB_HOST = os.getenv("MEMGRAPH_HOST", "localhost")
MEMGRAPH_URI = f"bolt://{DB_HOST}:7687"
AUTH = ("", "") 

#GPU SETUP
# Reduced batch size to prevent silent crashes - embedding takes approx 1HR on a 4090 RTX
BATCH_SIZE = 2048 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SNOMED-FINAL")
sys.stdout.reconfigure(line_buffering=True) # Force instant printing

try: csv.field_size_limit(sys.maxsize)
except OverflowError: csv.field_size_limit(2147483647) 

def check_files():
    """Verifies all files exist before starting."""
    missing = False
    for name, path in [("Concept", CONCEPT_FILE), ("Desc", DESC_FILE), ("Rel", REL_FILE)]:
        if not os.path.exists(path):
            logger.error(f"MISSING FILE: {name} at {path}")
            missing = True
        else:
            logger.info(f"Found {name}: {os.path.basename(path)}")
    if missing:
        logger.error("Aborting due to missing files.")
        sys.exit(1)

def load_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            yield row

def nuke_database(session):
    """Clears database and drops indexes safely."""
    logger.info("NUKING DATABASE...")
    session.run("MATCH (n) DETACH DELETE n")
    
    # Drop indexes one by one
    for idx in ["snomed_description_index", "concept_id_index"]:
        try: session.run(f"DROP INDEX {idx}") 
        except: pass
    try: session.run("DROP INDEX ON :Concept(sctid)") 
    except: pass
    try: session.run("DROP INDEX ON :Description(sctid)") 
    except: pass
    try: session.run("DROP VECTOR INDEX snomed_description_index")
    except: pass

def ingest_snomed():
    
    print(f"Using Device: {DEVICE.upper()}")
    if DEVICE == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    check_files()

    logger.info("Loading AI Model...")
    model = SentenceTransformer('all-mpnet-base-v2', device=DEVICE)
    logger.info("Model Loaded.")

    driver = GraphDatabase.driver(MEMGRAPH_URI, auth=AUTH)

    with driver.session() as session:
        nuke_database(session)
        logger.info("Creating Indexes...")
        session.run("CREATE INDEX ON :Concept(sctid)")
        session.run("CREATE INDEX ON :Description(sctid)")

    logger.info("Step 2: Loading Concepts...")
    with driver.session() as session:
        batch = []
        count = 0
        loaded_ids = set()
        
        for row in load_csv(CONCEPT_FILE):
            if row['active'] == '1':
                batch.append({"sctid": row['id']})
                loaded_ids.add(row['id'])
            
            if len(batch) >= BATCH_SIZE:
                session.run("UNWIND $batch as row MERGE (:Concept {sctid: row.sctid})", batch=batch)
                count += len(batch)
                batch = []
                print(f"   Saved {count} concepts...", end='\r')
        
        if batch:
            session.run("UNWIND $batch as row MERGE (:Concept {sctid: row.sctid})", batch=batch)
        print(f"Concepts Done. Total Active: {len(loaded_ids)}")

    logger.info("Step 3: Descriptions & Embeddings...")
    batch = []
    count = 0
    
    with driver.session() as session:
        for row in load_csv(DESC_FILE):
            if row['active'] == '1' and row['conceptId'] in loaded_ids:
                batch.append({
                    "id": row['id'],
                    "conceptId": row['conceptId'],
                    "term": row['term'],
                    "typeId": row['typeId'] 
                })

            if len(batch) >= BATCH_SIZE:
                try:
                   
                    terms = [item['term'] for item in batch]
                    vectors = model.encode(terms, batch_size=BATCH_SIZE, show_progress_bar=False).tolist()

                    for i, item in enumerate(batch):
                        item['embedding'] = vectors[i]

                    session.run("""
                        UNWIND $batch as row
                        MATCH (c:Concept {sctid: row.conceptId})
                        CREATE (d:Description {
                            sctid: row.id, 
                            term: row.term, 
                            type: row.typeId,
                            embedding: row.embedding
                        })
                        CREATE (c)-[:HAS_DESCRIPTION]->(d)
                    """, batch=batch)
                    
                    count += len(batch)
                    batch = []
                    print(f"Embedded & Saved {count} descriptions...", end='\r')


                    del terms
                    del vectors
                    # torch.cuda.empty_cache() 

                except Exception as e:
                    print(f"CRASH in Step 3: {e}")

                    sys.exit(1)

        # Flush final batch
        if batch:
            terms = [item['term'] for item in batch]
            vectors = model.encode(terms).tolist()
            for i, item in enumerate(batch):
                item['embedding'] = vectors[i]
            session.run("""
                UNWIND $batch as row
                MATCH (c:Concept {sctid: row.conceptId})
                CREATE (d:Description {sctid: row.id, term: row.term, type: row.typeId, embedding: row.embedding})
                CREATE (c)-[:HAS_DESCRIPTION]->(d)
            """, batch=batch)
            
    print(f"Descriptions Done: {count + len(batch)}")

    logger.info("Step 4: Relationships...")
    with driver.session() as session:
        batch_isa = []
        batch_assoc = []
        count = 0
        rels_map = {"116680003": "IS_A", "47429007": "ASSOCIATED_WITH", "42752001": "DUE_TO", "246090004": "ASSOCIATED_FINDING"}
        
        for row in load_csv(REL_FILE):
            if row['active'] == '1' and row['typeId'] in rels_map:
                if row['sourceId'] in loaded_ids and row['destinationId'] in loaded_ids:
                    item = {"source": row['sourceId'], "dest": row['destinationId']}
                    if row['typeId'] == "116680003":
                        batch_isa.append(item)
                    else:
                        batch_assoc.append(item)
            
            if len(batch_isa) >= BATCH_SIZE:
                session.run("UNWIND $batch as row MATCH (a:Concept {sctid: row.source}), (b:Concept {sctid: row.dest}) MERGE (a)-[:IS_A]->(b)", batch=batch_isa)
                count += len(batch_isa)
                batch_isa = []
                print(f"   Saved {count} relationships...", end='\r')

            if len(batch_assoc) >= BATCH_SIZE:
                session.run("UNWIND $batch as row MATCH (a:Concept {sctid: row.source}), (b:Concept {sctid: row.dest}) MERGE (a)-[:ASSOCIATED_WITH]->(b)", batch=batch_assoc)
                count += len(batch_assoc)
                batch_assoc = []
                print(f"   Saved {count} relationships...", end='\r')

        if batch_isa: session.run("UNWIND $batch as row MATCH (a:Concept {sctid: row.source}), (b:Concept {sctid: row.dest}) MERGE (a)-[:IS_A]->(b)", batch=batch_isa)
        if batch_assoc: session.run("UNWIND $batch as row MATCH (a:Concept {sctid: row.source}), (b:Concept {sctid: row.dest}) MERGE (a)-[:ASSOCIATED_WITH]->(b)", batch=batch_assoc)

    logger.info("Step 5: Building Vector Index...")
    with driver.session() as session:
        session.run("""
            CREATE VECTOR INDEX snomed_description_index ON :Description(embedding) 
            WITH CONFIG {"dimension": 768, "metric": "cos", "capacity": 1000000}
        """)
    logger.info("Done!")

if __name__ == "__main__":
    ingest_snomed()