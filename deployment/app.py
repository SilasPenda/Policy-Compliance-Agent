import gradio as gr
import requests
import json
import os
import tempfile
import uuid
from dotenv import load_dotenv

from agent.templates import parser
from agent.reasoning import create_compliance_agent
from src.utils import read_pdf, redis_client_connect

# Flask API endpoint
load_dotenv()

compliance_url = os.getenv("COMPLIANCE_URL")

def process_query(session_id, pdf_text, query):
    try:
        data = {"session_id": session_id, "pdf_text": pdf_text, "query": query}
        response = requests.post(compliance_url, json=data)
            
        if response.status_code != 200:
            return f"API Error [{response.status_code}]: {response.text}"

        result = response.json()

        if "error" in result:
            return f"API returned an error: {result['error']}"

        # Format the response for Gradio output
        verdict = result.get("verdict", "Unknown")
        reasoning = result.get("reasoning", "")
        confidence = result.get("confidence", 0.0)

        if verdict == "Compliant":
            policies = result.get("compliant_policies", [])
            policies_text = "\n".join(f"- {p}" for p in policies)
        else:
            policies = result.get("violated_policies", [])
            policies_text = "\n".join(f"- {p}" for p in policies)

        tools_text = "\n".join(f"- {t}" for t in result.get("tools_used", []))
        similar_docs_text = "\n".join(f"- {d}" for d in result.get("similar_documents", []))

        return f"""**Compliance Status:** {verdict}
        
                **Policies:** 
                {policies_text}

                **Tools Used:**
                {tools_text}

                **Similar Documents:**
                {similar_docs_text}

                **Reasoning:**
                {reasoning}

                **Confidence:** {confidence:.2f}
                """

    except Exception as e:
        return f"Error contacting API: {e}"

def load_pdf(pdf_file):
    if pdf_file is None:
        return "", "No file uploaded."
    
    # Get the file name
    file_name = pdf_file.name.split("/")[-1]  # Grabs only the file name, not full path
    
    document_pages = read_pdf(pdf_file.name)
    texts = [page.extract_text() for page in document_pages if page.extract_text()]
    full_text = "\n\n".join(texts)
    return full_text, file_name


with gr.Blocks() as app:
    gr.Markdown("## 📘 Policy Compliance Assistant")

    with gr.Row():
        pdf_input = gr.File(label="Upload a PDF", file_types=[".pdf", ".PDF"])
        query_input = gr.Textbox(label="Compliance Question", placeholder="Enter your compliance question here...", lines=2)

    pdf_text_state = gr.State(value="")  # stores PDF text after upload
    file_name_state = gr.State(value="")  # stores the file name
    output = gr.Textbox(label="Compliance Report", lines=20, interactive=False)
    run_btn = gr.Button("Run Compliance Audit")

    # Load PDF once when uploaded
    pdf_input.upload(fn=load_pdf, inputs=pdf_input, outputs=[pdf_text_state, file_name_state])

    session_id_state = gr.State(value=str(uuid.uuid4()))

    # Run query using stored PDF text
    run_btn.click(fn=process_query, inputs=[session_id_state, pdf_text_state, query_input], outputs=output)

app.launch(pwa=True, server_name="0.0.0.0", server_port=8501)