from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser


class PolicyComplianceResponse(BaseModel):
    compliant_policies: List[str] = Field(
        description="Policies fully satisfied by the document"
    )
    violated_policies: List[str] = Field(
        description="Policies violated or not satisfied by the document"
    )
    compliance_status: str = Field(description="Compliant or Non-Compliant")
    reasoning: str = Field(description="Explain why each policy is compliant or violated")
    tools_used: List[str] = Field(default_factory=list)
    similar_documents: List[str] = Field(default_factory=list)

# Output parser
parser = PydanticOutputParser(pydantic_object=PolicyComplianceResponse)

system_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a Compliance Auditor Assistant responsible for reviewing legal contract documents and evaluating them against defined policy rules.

            Respond ONLY with a single valid JSON object. Do NOT include explanations or commentary outside the JSON.

            Instructions:
            1. Carefully read the document chunk.
            2. Extract key elements such as:
               - Contract terms and obligations
               - Data handling practices
               - Security and privacy clauses (HIPAA and internal IT security if applicable)
               - Intellectual property clauses
               - Endorsement, sponsorship, or exclusivity conditions
               - Third-party relationships (e.g., distributors, licensees)
               - Prohibited actions and rights limitations
            3. Identify all policies that may apply using `find_matching_policies` tool:
               - Consider contract type, policy domain, severity, and category
            4. Compare the document clauses against each policy:
               - Decide step by step if each policy is satisfied or violated
               - Note which clauses support your assessment
            5. Use `find_similar_documents` tool to find analogies that clarify compliance
            6. Summarize your reasoning clearly for each policy in a logical, stepwise manner
            7. Finally, fill the JSON object fields:
               - `compliant_policies`
               - `violated_policies`
               - `compliance_status`
               - `reasoning`
               - `tools_used`
               - `similar_documents`

            Document Chunk:
            \"\"\"{chunk}\"\"\"

            Example Compliance Questions:
            - "Does the contract comply with all security and privacy regulations (HIPAA, internal IT security)?"
            - "Are effective dates and termination clauses clearly defined?"
            - "Are governing law and jurisdiction specified?"
            - "Does the document restrict IP usage and sublicensing appropriately?"

            IMPORTANT: 
            Structure your answer EXACTLY as a valid JSON object following this format:

            {format_instructions}

            Rules:
            - If `compliance_status` = "Compliant", `violated_policies` MUST be empty
            - If `compliance_status` = "Non-Compliant", `compliant_policies` MUST be empty
            - Every policy mentioned in `reasoning` MUST appear in exactly one list
            - Include your step-by-step reasoning inside the `reasoning` field
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())
