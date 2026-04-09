# Policy_Compliance_Agent

A smart AI-powered Compliance Auditor Assistant designed to analyze documents, extract relevant compliance information, and determine whether they meet specified policy standards such as GDPR. This tool leverages embeddings, similarity search, and LLMs to provide detailed compliance assessments based on custom policies and similar documents.

---

## Features

- **Document Chunk Analysis:** Reads and analyzes PDF document chunks for compliance-relevant entities.
- **Policy Matching:** Retrieves and cross-checks document content against relevant policy rules using semantic search.
- **Similar Document Reference:** Finds and compares similar documents to strengthen audit decisions.
- **Compliance Decision:** Outputs a clear **Compliant** or **Non-Compliant** status with detailed explanations.
- **Extensible Tools:** Easily integrate additional policy collections and compliance questions.
- **Gradio UI:** User-friendly interface for uploading documents, entering queries, and viewing compliance reports.
- **Dockerized:** Ready for containerized deployment.

---

## Architecture

- **Embeddings & Vector Search:** Uses a vector database (e.g., ChromaDB) to store and query policy rules and similar documents efficiently.
- **Large Language Model (LLM):** Applies an LLM to reason about compliance based on the context, retrieved policies, and related documents.
- **Gradio Frontend:** Provides an interactive web UI for document upload and compliance queries.
- **Modular Tools:** Custom tools like `find_matching_policies` and `find_similar_documents` encapsulate business logic.

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- Access to OpenAI API or other LLM providers like Ollama
- Vector database setup (e.g., ChromaDB)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SilasPenda/Policy-Compliance-Agent
   cd policy-compliance-auditor

2. Create & activate virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate (Linux & Mac)
   ./.venv/Scripts/activate (Windows)
   
3. Install requirements:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt

4. Create .env file and add OPEN_AI_KEY

5. Create collections for contracts:
   ```bash
      python ingestion/contract_ingestor.py

6. Create collections for policies:
   ```bash
      python ingestion/policy_ingestor.py

7. Launch API

   ```bash
   uvicorn deployment.api:app --reload

8. Start App

   ```bash
   python deployment/app.py
