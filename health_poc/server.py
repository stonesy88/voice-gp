import os
import logging
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Triage-FastAPI")

# Database Connection
DB_HOST = os.getenv("MEMGRAPH_HOST", "health_memgraph")
MEMGRAPH_URI = f"bolt://{DB_HOST}:7687"
AUTH = ("", "") 

# --- LOAD AI MODEL ---
logger.info("Loading AI Model...")
device = 'cuda' if os.getenv("USE_GPU", "false").lower() == "true" else 'cpu'
try:
    model = SentenceTransformer('all-mpnet-base-v2', device=device)
    logger.info(f"Model Ready on {device.upper()}.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    exit(1)

driver = GraphDatabase.driver(MEMGRAPH_URI, auth=AUTH)

# --- TOOLS ---
def lookup_symptom(symptom_text):
    """Vector search using sync driver (FastAPI puts this in a threadpool automatically)"""
    logger.info(f"Searching for: {symptom_text}")
    try:
        embedding = model.encode(symptom_text).tolist()
        query = """
            CALL vector_search.search('snomed_description_index', 5, $vec) 
            YIELD node, score
            MATCH (c:Concept)-[:HAS_DESCRIPTION]->(node)
            RETURN score, node.term AS term, c.sctid AS id
        """
        with driver.session() as session:
            result = session.run(query, embedding=embedding)
            candidates = [{"id": r["id"], "term": r["term"], "score": r["score"]} for r in result]

            if candidates:
            top_matches = [f"{c['term']} ({c['score']:.2f})" for c in candidates[:3]]
            logger.info(f"Results: {top_matches} ...")
        else:
            logger.warning("No matches found.")

        return candidates
    except Exception as e:
        logger.error(f"DB Query Failed: {e}")
        return []

# --- WEBHOOK ---
@app.post("/triage")
async def triage_webhook(request: Request):
    """Handle Vapi function calls"""
    try:
        data = await request.json()
        message = data.get('message', {})
        
        if message.get('type') == 'function-call':
            function_call = message.get('functionCall', {})
            name = function_call.get('name')
            params = function_call.get('parameters', {})
            
            logger.info(f"Tool Call: {name} | Params: {params}")

            if name == 'lookup_symptom':
                result = lookup_symptom(params.get('symptom'))
                return JSONResponse(content={
                    "results": [{
                        "toolCallId": function_call.get('toolCallId'),
                        "result": str(result)
                    }]
                })

        return JSONResponse(content={"status": "ok"})

    except Exception as e:
        logger.error(f"Webhook Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)