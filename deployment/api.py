import json
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.utils import get_embedding_model, read_pdf, compute_confidence, redis_client_connect
from src.schemas import Compliance, HistoryResponse, RetrieveHistory
from agent.templates import parser
from agent.reasoning import create_compliance_agent

from dotenv import load_dotenv
load_dotenv()



app = FastAPI(title="Policy Compliance API")

app = FastAPI(
    title="Policy Compliance API",
    description="This is the first version of Policy Compliance API",
    version="1.0.0"
)

redis = redis_client_connect()

# --- Health check ---
@app.get("/health")
async def health():
    return {"status": "ok"}

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def find_similar_query_embedding(new_emb, session_id, threshold=0.88):
    """
    Check Redis for similar queries in the same session.
    Returns the cached response if similarity >= threshold.
    """
    keys = redis.lrange(session_id, 0, -1)
    for key_bytes in keys:
        key_data = json.loads(key_bytes)
        past_emb = key_data.get("embedding")
        if past_emb:
            similarity = cosine_similarity(np.array(new_emb), np.array(past_emb))
            if similarity >= threshold:
                return key_data["response"]
    return None

@app.post("/compliance/check")
async def compliance_check(check_info: Compliance):
    try:
        # --- Step 1: Compute embedding for new query ---
        embedding_model = get_embedding_model()

        query_embedding = embedding_model.encode(check_info.query).tolist()

        # --- Step 2: Check Redis for a semantically similar previous query ---
        existing_response = None
        existing_response = find_similar_query_embedding(query_embedding, check_info.session_id)

        if existing_response:
            # Found similar query, return cached response
            return JSONResponse(content=existing_response)

        # --- Step 3: No similar query, call the agent ---
        agent_executor = create_compliance_agent(llm_type="openai", model_name="gpt-4o")

        # Invoke agent
        response = agent_executor.invoke({
            "query": check_info.query,
            "chunk": check_info.pdf_text,
            "agent_scratchpad": ""
        })

        structured_response = parser.parse(response.get("output"))
        confidence_score = compute_confidence(structured_response)

        # --- Step 4: Store response + embedding in Redis ---
        entry = {
            "query": check_info.query,
            "embedding": query_embedding,
            "response": {
                "verdict": structured_response.compliance_status,
                "compliant_policies": structured_response.compliant_policies,
                "violated_policies": structured_response.violated_policies,
                "tools_used": structured_response.tools_used,
                "similar_documents": structured_response.similar_documents,
                "reasoning": structured_response.reasoning,
                "confidence": confidence_score,
            }
        }
        redis.rpush(check_info.session_id, json.dumps(entry))
        redis.expire(check_info.session_id, check_info.ttl)

        # --- Step 5: Return structured JSON ---
        return JSONResponse(content={
            "verdict": structured_response.compliance_status,
            "compliant_policies": structured_response.compliant_policies,
            "violated_policies": structured_response.violated_policies,
            "tools_used": structured_response.tools_used,
            "similar_documents": structured_response.similar_documents,
            "reasoning": structured_response.reasoning,
            "confidence": confidence_score,
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "verdict": "unknown",
                "compliant_policies": [],
                "violated_policies": [],
                "tools_used": [],
                "similar_documents": [],
                "reasoning": "",
                "confidence": 0.0,
                "error": str(e),
            }
        )
