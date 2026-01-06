import os
import json
import logging
from fastapi import FastAPI, Request
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 1. Load Model (Must match ingest.py)
logger.info("Loading Embedding Model...")
model = SentenceTransformer('all-mpnet-base-v2')
logger.info("Model Loaded.")

# 2. Database Connection
# If running inside Docker, use "bolt://memgraph:7687"
# If running locally with ngrok, use "bolt://localhost:7687"
MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://memgraph:7687")
AUTH = ("", "")
driver = GraphDatabase.driver(MEMGRAPH_URI, auth=AUTH)

def get_embedding(text):
    return model.encode(text).tolist()

@app.post("/vapi-webhook")
async def vapi_webhook(request: Request):
    payload = await request.json()
    
    # Extract Tool Call
    message = payload.get("message", {})
    tool_calls = message.get("toolCalls", [])
    
    if not tool_calls:
        return {"results": []}

    tool_call = tool_calls[0]
    function_name = tool_call["function"]["name"]
    call_id = tool_call["id"]
    
    logger.info(f"Received Tool Call: {function_name}")
    
    result_text = "Error processing request."

    if function_name == "lookup_medical_graph":
        try:
            args = json.loads(tool_call["function"]["arguments"])
            symptom = args.get("symptom", "")
            
            logger.info(f"Looking up symptom: {symptom}")
            
            # 1. Vector Search
            vector = get_embedding(symptom)
            
            # Query Logic:
            # - Find the closest Description node by vector similarity
            # - Find the Concept owning that Description
            # - Find Conditions associated with that Concept
            query = """
            CALL vector_search.search('snomed_description_index', $embedding, 1) 
            YIELD node AS matched_desc, similarity
            
            MATCH (matched_desc)<-[:HAS_DESCRIPTION]-(symptom_concept:Concept)
            MATCH (symptom_concept)<-[:ASSOCIATED_WITH]-(condition:Concept)
            MATCH (condition)-[:HAS_DESCRIPTION]->(cond_desc:Description {type: 'FSN'})
            
            RETURN 
                matched_desc.term AS Identified_Symptom,
                similarity AS Score,
                collect(cond_desc.term) AS Potential_Conditions
            """
            
            with driver.session() as session:
                data = session.run(query, embedding=vector).data()
                
            if data:
                top_match = data[0]
                found_symptom = top_match['Identified_Symptom']
                conditions = top_match['Potential_Conditions']
                
                result_text = (
                    f"I found a match for '{found_symptom}' in the clinical database. "
                    f"This is often associated with: {', '.join(conditions)}. "
                    "Please ask differentiating questions."
                )
            else:
                result_text = "I could not find a specific clinical match for that symptom."

        except Exception as e:
            logger.error(f"Error in graph lookup: {e}")
            result_text = "I am having trouble accessing the records right now."

    return {
        "results": [
            {
                "toolCallId": call_id,
                "result": result_text
            }
        ]
    }