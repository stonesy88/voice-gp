import os
import logging
import json
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

# --- SEARCH LOGIC ---
def lookup_symptom(symptom_text):
    """Generates embedding and searches Memgraph"""
    logger.info(f"🔎 Embedding and searching for: '{symptom_text}'")
    try:
        # 1. Generate Embedding
        embedding = model.encode(symptom_text).tolist()
        
        # 2. Search Memgraph (Using MAGE vector_search)
        query = """
        CALL vector_search.search('snomed_description_index', 10, $embedding) 
        YIELD node, similarity 
        MATCH (c:Concept)-[:HAS_DESCRIPTION]->(node)
        RETURN c.sctid AS id, node.term AS term, similarity AS score 
        ORDER BY score DESC
        """
        
        with driver.session() as session:
            result = session.run(query, embedding=embedding)
            # Parse results into a clean list
            candidates = [
                {"id": r["id"], "term": r["term"], "score": float(r["score"])} 
                for r in result
            ]

            if candidates:
                top = candidates[0]
                logger.info(f"Top Match: {top['term']} ({top['score']:.4f})")
            else:
                logger.warning("No results returned from DB.")

        return candidates

    except Exception as e:
        logger.error(f"DB Query Failed: {e}")
        traceback.print_exc()
        return []

# --- WEBHOOK HANDLER (FIXED) ---
@app.post("/triage")
async def triage_webhook(request: Request):
    """Handle Vapi Tool Calls"""
    try:
        data = await request.json()
        message = data.get('message', {})
        
        # 1. DETECT TOOL CALLS (New Vapi/OpenAI Format)
        tool_calls = message.get('toolCalls', [])
        
        if not tool_calls:
            # Fallback for manual testing via Postman
            if data.get('symptom'):
                logger.info("Manual Postman/Curl Request Detected")
                result = lookup_symptom(data['symptom'])
                return JSONResponse(content={"results": result})
            
            logger.info("Received heartbeat/status update (no tool calls).")
            return JSONResponse(content={"status": "ok"})

        # 2. PROCESS FIRST TOOL CALL
        call = tool_calls[0]
        call_id = call.get('id')
        function = call.get('function', {})
        name = function.get('name')
        
        # 3. PARSE ARGUMENTS (Can be dict or JSON string)
        args_raw = function.get('arguments', {})
        if isinstance(args_raw, str):
            args = json.loads(args_raw)
        else:
            args = args_raw
            
        logger.info(f"Tool Call Detected: {name} | ID: {call_id}")

        # 4. EXECUTE & RESPOND
        response_data = {}
        if name == 'lookup_symptom':
            symptom = args.get('symptom')
            if symptom:
                search_results = lookup_symptom(symptom)
                
                # Format exactly as Vapi expects
                response_data = {
                    "results": [
                        {
                            "toolCallId": call_id,
                            "result": json.dumps(search_results)  # LLM needs stringified JSON
                        }
                    ]
                }
            else:
                logger.error("Tool call missing 'symptom' argument.")

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Webhook Error: {e}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)