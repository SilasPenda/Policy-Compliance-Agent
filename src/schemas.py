from typing import List, Optional
from pydantic import BaseModel, Field

        
class Compliance(BaseModel):
    session_id : str = Field(...)
    pdf_text: str = Field(...)
    query: str = Field(...)
    # history: Optional[List[dict]] = Field(default_factory=list)
    ttl: int = Field(default=86400)  # Default TTL of 24 hours for history entries
    
    class Config:
        allowed_population_by_field_name = True
        arbitrary_types_allowed = True
        schema_extra = {
            "example" : {
                "session_id": "64b8f0c2e1d3f2a5b6c7d8e",
                "pdf_text": "This is the extracted text from the uploaded PDF document...",
                "query": "Is this document compliant with GDPR regulations?",
                "ttl": 86400
            }
        }

# class CreateHistory(BaseModel):
#     session_id : str = Field(...)
#     query: str = Field(...)
#     response: str = Field(...)
#     ttl: int = Field(default=86400)  # Default TTL of 24 hours for history entries
    
#     class Config:
#         allowed_population_by_field_name = True
#         arbitrary_types_allowed = True
#         schema_extra = {
#             "example" : {
#                 "session_id": "64b8f0c2e1d3f2a5b6c7d8e",
#                 "query": "Is this document compliant with GDPR regulations?",
#                 "response": "The document is compliant with GDPR regulations because...",
#                 "ttl": 3600
#             }
#         }

class HistoryResponse(BaseModel):
    history: list = Field(...)

    class Config:
        allowed_population_by_field_name = True
        arbitrary_types_allowed = True
        schema_extra = {
            "example" : {
                "history": [
                    {"query": "Is this document compliant with GDPR regulations?", "response": {
                        "verdict": "Compliant",
                        "reasoning": "The document meets all the necessary requirements outlined in GDPR.",
                        "confidence": 0.95,
                        "compliant_policies": ["Data Protection", "User Consent"],
                        "tools_used": ["Policy Checker", "Compliance Analyzer"],
                        "similar_documents": ["Document A", "Document B"]
                    }}
                ]
            }
        }           

class RetrieveHistory(BaseModel):
    session_id : str = Field(...)

    class Config:
        allowed_population_by_field_name = True
        arbitrary_types_allowed = True
        schema_extra = {
            "example" : {
                "session_id": "64b8f0c2e1d3f2a5b6c7d8e"
            }
        }


    