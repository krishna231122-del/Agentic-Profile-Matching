import os
import json
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass, field

from datetime import datetime
import operator
from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
import chromadb 
 ## llm implementation 
from dotenv import load_dotenv
load_dotenv()

# Load environment variables from .env file
llm = ChatMistralAI(
    model="mistral-small-2603",
    api_key= os.getenv("MISTRAL_API_KEY")   
)
## building agenti state which also store chat history 

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    job_description: str
    requirements: Dict[str, Any]          # {"must_have": [...], "nice_to_have": [...]}
    candidate_pool: List[Dict[str, Any]]  # raw resumes fetched from file system / RAG
    shortlist: List[Dict[str, Any]]       # top-N candidates with scores + reasoning
    ranking_reasoning: str
    round_number: int                     # 1 = initial screen, 2 = deep analysis, 3 = final
    final_report: str
    human_feedback: Optional[str]
    awaiting_feedback: bool
    refinement_history: List[Dict[str, Any]]
    summary : str

# tools implementation 

RESUME_DIR = "./Resume_dir"

Embeddings = MistralAIEmbeddings(
    model="mistral-embed-2312",
    api_key=os.getenv("MISTRAL_API_KEY"))


VECTOR_DB_DIR = "./chroma_resume_db"

vector_store = Chroma(
    collection_name="resumes",
    embedding_function=Embeddings,
    persist_directory=VECTOR_DB_DIR,
)


def index_resumes(resume_dir: str = RESUME_DIR):
    """
    Reads resumes -> chunks -> embeds (Mistral) -> stores in Chroma.
    Skips empty/whitespace chunks to avoid Mistral API 400 errors.
    """
    if not os.path.exists(resume_dir):
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = []
    skipped_files = []
    skipped_chunks = 0

    for filename in os.listdir(resume_dir):
        if not filename.endswith((".txt", ".pdf", ".docx")):
            continue

        text = read__resume.invoke({"filename": filename})

        # Skip files that failed to read or returned nothing useful
        if not text or not text.strip() or text.startswith("Error:"):
            skipped_files.append(filename)
            continue

        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:                      # <-- prevents the Mistral 400 error
                skipped_chunks += 1
                continue
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "candidate_id": filename,
                    "filename": filename,
                    "chunk_index": i,
                },
            ))

    if skipped_files:
        print(f"⚠️ Skipped {len(skipped_files)} unreadable files: {skipped_files}")
    if skipped_chunks:
        print(f"⚠️ Skipped {skipped_chunks} empty chunks")

    if not docs:
        raise ValueError(
            "No valid text chunks found to index. "
            "Check that your resume files contain readable text."
        )

    # Embed and add in small batches to isolate failures more easily
    batch_size = 20
    total_added = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        try:
            vector_store.add_documents(batch)
            total_added += len(batch)
        except Exception as e:
            print(f"❌ Failed embedding batch {i}-{i+len(batch)}: {e}")
            # Print the offending content for debugging
            for d in batch:
                print(f"   -> {d.metadata['filename']} chunk {d.metadata['chunk_index']}: "
                      f"{repr(d.page_content[:50])}")
            raise

    return f"✅ Indexed {total_added} chunks from {len(os.listdir(resume_dir))} files in {resume_dir}"


# ---------------------------------------------------------------------------
# RAG SEARCH TOOL
# ---------------------------------------------------------------------------

@tool
def rag_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Semantic search over the resume vector store using Mistral embeddings.
    Returns top_k candidates matching the query with similarity scores.
    """
    if not query or not query.strip():
        return []

    results = vector_store.similarity_search_with_relevance_scores(query, k=top_k)

    candidates: Dict[str, Dict[str, Any]] = {}
    for doc, score in results:
        cid = doc.metadata["candidate_id"]
        if cid not in candidates or score > candidates[cid]["similarity_score"]:
            candidates[cid] = {
                "candidate_id": cid,
                "name": doc.metadata.get("filename", cid),
                "resume_text": doc.page_content,
                "similarity_score": round(score, 3),
            }

    return sorted(candidates.values(), key=lambda c: c["similarity_score"], reverse=True)[:top_k]



@tool
def list_resumes() -> List[str]:
    """List all resume filenames available in the resume directory."""
    if not os.path.exists(RESUME_DIR):
        return []
    return [f for f in os.listdir(RESUME_DIR) if f.endswith((".pdf", ".txt", ".docx"))]


# only txt file saportes 
@tool
def read__resume(filename: str) -> str:
    """Read a resume file and return its content as a string """
    path = os.path.join(RESUME_DIR, filename)
    if not os.path.exists(path):
            raise FileNotFoundError(f"Resume file {filename} not found in {RESUME_DIR}")
    if filename.endswith(".txt"):
         with open (path,"r" , encoding="utf-8") as f:
          return f.read()
         
         return f"extracted text from {filename}"  
    

## save tool
@tool
def save_resume(filename: str , content: str)-> str:
    """ save the generated resume content to a file """
    os.makedirs(RESUME_DIR, exist_ok=True)
    path = os.path.join(RESUME_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Resume saved to {filename}"


@tool
def extract_requirements(jd:str)-> dict[str,List[str]]:
    """ Parse a job description into must-have vs nice-to-have requirements."""
    prompt = f"""Analyze this job description and extract requirements as JSON with keys
"must_have" and "nice_to_have", each a list of short requirement strings
(skills, years of experience, education, certifications).

    
job_discription :
{jd}
Respond ONLY with valid JSON, no markdown fences."""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
               return json.loads(response.content)
    
    except json.JSONDecodeError:
           return {"must_have": [], "nice_to_have": [], "raw": response.content}
    

@tool
def compare_candidates(candidate_ids: List[str], candidate_data: Optional[Dict[str, Any]] = None) -> str:
    """Perform a head-to-head comparison of multiple candidates."""
    candidates_info = candidate_data or {}
    prompt = f"""Compare these candidates head-to-head for the role.
Candidate IDs: {candidate_ids}
Candidate Data: {json.dumps(candidates_info, indent=2)}

Provide:
1. A comparison table (skills, experience, education)
2. Relative strengths/weaknesses of each
3. A recommendation on ranking order with justification."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def generate_interview_questions(candidate_id: str, candidate_summary: str = "") -> List[str]:
    """Generate tailored screening interview questions for a specific candidate."""
    prompt = f"""Generate 8 targeted interview screening questions for candidate {candidate_id}.
Candidate summary: {candidate_summary}

Include:
- 3 technical/skill-verification questions
- 2 experience-depth questions
- 1 behavioral question
- 1 DSA question 
- 1 gap-probing question (if any resume gaps/weaknesses exist)

Return as a numbered list."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return [line.strip() for line in response.content.split("\n") if line.strip()]

ALL_TOOLS = [
    list_resumes,
    read__resume,
    save_resume,
    rag_search,
    extract_requirements,
    compare_candidates,
    generate_interview_questions,
]

llm_with_tools = llm.bind_tools(ALL_TOOLS)


## defining node 

def jd_parser_node(state: AgentState) -> AgentState:
    """START -> Parse JD"""
    jd = state["job_description"]
    msg = AIMessage(content=f"Parsing job description ({len(jd)} chars)...")
    return {"messages": [msg]}

def extract_requirements_node(state: AgentState) -> AgentState:
    """Extract Requirements"""
    reqs = extract_requirements.invoke({"jd": state["job_description"]})
    summary = (
        f"**Must-have:** {', '.join(reqs.get('must_have', []))}\n"
        f"**Nice-to-have:** {', '.join(reqs.get('nice_to_have', []))}"
    )
    return {
        "requirements": reqs,
        "messages": [AIMessage(content=f"Extracted requirements:\n{summary}")],
    }

def search_resume_node(state:AgentState)->AgentState:
     """search resumes using RAG"""
     reqs = state["requirements"]
     query = " ".join(reqs.get("must_have", []) + reqs.get("nice_to_have", []))
     results =rag_search.invoke({"query": query, "top_k": 3})
     return {
         "candidate_pool": results,     
         "messages": [AIMessage(content=f"Found {len(results)} candidates matching requirements.")]
     }

def Rank_candidates_node(state:AgentState)->AgentState:
     """Rank Candidates - multi-round screening"""
     round_num = state.get("round_number", 1)
     pool = state.get("candidate_pool")
     reqs = state.get("requirements")

     if round_num == 1:
          # Round 1: initial screen -> top 10 from full pool
        target_n = 3
        instruction = "Perform an initial screen and select the top candidates."

     elif round_num==2:
          # Round 2: deep analysis -> top 5 from round 1 shortlist
        target_n = 2
        instruction = "Perform deep analysis: verify skill depth, experience relevance, and career trajectory."

     else:
          # Round 3: final selection -> top 3 from round 2 shortlist
        target_n = 2
        instruction = "Perform final selection based on behavioral and nature of work."

     prompt = f"""{instruction}

Job Requirements:
Must-have: {reqs.get('must_have')}
Nice-to-have: {reqs.get('nice_to_have')}

Candidates:
{json.dumps(pool, indent=2)}
Return JSON list of up to {target_n} objects, each with:
- candidate_id
- name
- score (0-100)
- strengths (list)
- gaps (list)
- reasoning (string)
{"- recommendation ('hire'/'no-hire'/'borderline') and improvement_suggestions (if borderline)" if round_num == 3 else ""}

Sort by score descending. Respond ONLY with valid JSON."""

     response = llm.invoke([HumanMessage(content=prompt)])
     try:
        ranked = json.loads(response.content)
     except json.JSONDecodeError:
        ranked = [{"raw_output": response.content}]

     reasoning_text = f"Round {round_num} ranking complete. {len(ranked)} candidates ranked."

     return {
        "shortlist": ranked,
        "ranking_reasoning": reasoning_text,
        "round_number": round_num + 1,
        "messages": [AIMessage(content=reasoning_text)],
     }

def generate_report_node(state: AgentState) -> AgentState:
    """Generate Report - explainability"""
    shortlist = state["shortlist"]
    reqs = state["requirements"]

    prompt = f"""Generate a detailed candidate match report.

Requirements: {json.dumps(reqs)}
Ranked candidates: {json.dumps(shortlist, indent=2)}

For each candidate include:
1. Overall match score and rank
2. Key strengths (mapped to requirements)
3. Gaps or concerns
4. For borderline candidates: specific improvement suggestions
5. A one-line hire recommendation summary

Format as clean markdown."""

    response = llm.invoke([HumanMessage(content=prompt)])
    report = response.content

    return {
        "final_report": report,
        "messages": [AIMessage(content=report)],
    }

def human_feedback_node(state: AgentState) -> AgentState:
    """Human Feedback Loop - pause for user input"""
    return {
        "awaiting_feedback": True,
        "messages": [AIMessage(
            content="Report generated. Please review and provide feedback "
                    "(e.g., adjust requirements, ask for comparisons, request interview "
                    "questions, or approve/finalize)."
        )],
    }

def route_after_feedback(state: AgentState) -> str:
    feedback = (state.get("human_feedback") or "").lower()

    if not feedback:
        return "end"

    if any(
        k in feedback
        for k in [
            "adjust",
            "change",
            "modify",
            "update",
            "recommendation",
            "role",
            "data engineer",
            "software engineer",
            "refine",
        ]
    ):
        return "refine_requirements"

    if any(k in feedback for k in ["compare", "vs", "versus"]):
        return "compare"

    if any(k in feedback for k in ["interview", "screening question"]):
        return "interview_questions"

    if any(k in feedback for k in ["next round", "deep analysis", "final round"]):
        return "next_round"

    if any(k in feedback for k in ["approve", "done", "finalize"]):
        return "end"

    return "conversational_response"

def refine_requirements_node(state: AgentState) -> AgentState:
    """Iterative Refinement - adjust requirements mid-conversation"""
    feedback = state["human_feedback"]
    current_reqs = state["requirements"]

    prompt = f"""Current requirements: {json.dumps(current_reqs)}
User feedback / refinement request: "{feedback}"

Update the requirements JSON accordingly (keys: must_have, nice_to_have).
Respond ONLY with the updated valid JSON."""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        new_reqs = json.loads(response.content)
    except json.JSONDecodeError:
        new_reqs = current_reqs

    refinement_entry = {
        "timestamp": datetime.now().isoformat(),
        "feedback": feedback,
        "old_requirements": current_reqs,
        "new_requirements": new_reqs,
    }

    history = state.get("refinement_history", [])
    history.append(refinement_entry)

    explanation = (
        f"Updated requirements based on your feedback: '{feedback}'.\n"
        f"New must-have: {new_reqs.get('must_have')}\n"
        f"New nice-to-have: {new_reqs.get('nice_to_have')}\n"
        f"Re-ranking candidates now..."
    )

    return {
        "requirements": new_reqs,
        "refinement_history": history,
        "awaiting_feedback": False,
        "human_feedback": None,
        "messages": [AIMessage(content=explanation)],
    }


def compare_node(state: AgentState) -> AgentState:
    """Handle comparison requests ('Why did John rank higher than Jane?')"""
    feedback = state["human_feedback"]
    shortlist = state["shortlist"]

    prompt = f"""User asked: "{feedback}"

Shortlist with scores/reasoning: {json.dumps(shortlist, indent=2)}

Answer the user's comparison question directly and specifically,
citing concrete evidence from each candidate's scoring/reasoning."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "awaiting_feedback": False,
        "human_feedback": None,
        "messages": [AIMessage(content=response.content)],
    }


def interview_questions_node(state: AgentState) -> AgentState:
    """Generate interview questions for a requested candidate."""
    feedback = state["human_feedback"]
    shortlist = state["shortlist"]

    id_prompt = f"""User asked: "{feedback}"
Shortlist: {json.dumps([c.get('candidate_id') for c in shortlist if isinstance(c, dict)])}
Return ONLY the candidate_id being referenced."""
    id_response = llm.invoke([HumanMessage(content=id_prompt)])
    candidate_id = id_response.content.strip()

    candidate_summary = next(
        (json.dumps(c) for c in shortlist if isinstance(c, dict) and c.get("candidate_id") == candidate_id),
        ""
    )
    questions = generate_interview_questions.invoke({
        "candidate_id": candidate_id,
        "candidate_summary": candidate_summary,
    })

    return {
        "awaiting_feedback": False,
        "human_feedback": None,
        "messages": [AIMessage(content=f"Interview questions for {candidate_id}:\n" + "\n".join(questions))],
    }


def conversational_node(state: AgentState) -> AgentState:
    """Generic conversational fallback using tool-bound LLM."""
    feedback = state["human_feedback"]
    context_msg = HumanMessage(content=feedback)
    response = llm.invoke(state["messages"] + [context_msg])
    return {
        "awaiting_feedback": False,
        "human_feedback": None,
        "messages": [context_msg, response],
    }


def route_after_refine_or_round(state: AgentState) -> str:
    """After refinement or next-round request, decide whether to re-rank."""
    return "rank_candidates"


# ---------------------------------------------------------------------------
# BUILD GRAPH
# ---------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_jd", jd_parser_node)
    workflow.add_node("extract_requirements", extract_requirements_node)
    workflow.add_node("search_resumes", search_resume_node)
    workflow.add_node("rank_candidates", Rank_candidates_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("human_feedback", human_feedback_node)
    workflow.add_node("refine_requirements", refine_requirements_node)
    workflow.add_node("compare", compare_node)
    workflow.add_node("interview_questions", interview_questions_node)
    workflow.add_node("conversational_response", conversational_node)

    workflow.set_entry_point("parse_jd")

    workflow.add_edge("parse_jd", "extract_requirements")
    workflow.add_edge("extract_requirements", "search_resumes")
    workflow.add_edge("search_resumes", "rank_candidates")
    workflow.add_edge("rank_candidates", "generate_report")
    workflow.add_edge("generate_report", "human_feedback")

    workflow.add_conditional_edges(
        "human_feedback",
        route_after_feedback,
        {
            "refine_requirements": "refine_requirements",
            "compare": "compare",
            "interview_questions": "interview_questions",
            "next_round": "rank_candidates",
            "conversational_response": "conversational_response",
            "end": END,
        },
    )

    workflow.add_edge("refine_requirements", "rank_candidates")
    workflow.add_edge("compare", "human_feedback")
    workflow.add_edge("interview_questions", "human_feedback")
    workflow.add_edge("conversational_response", "human_feedback")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["human_feedback"])


# ---------------------------------------------------------------------------
# AGENT WRAPPER
# ---------------------------------------------------------------------------

class MatchingAgent:
    """High-level interface for the recruiting matching agent."""

    def __init__(self):
        self.graph = build_graph()
        self.thread_id = f"session_{datetime.now().timestamp()}"
        self.config = {"configurable": {"thread_id": self.thread_id}}

    def start_screening(self, job_description: str, round_number: int = 1) -> Dict[str, Any]:
        initial_state = {
            "messages": [SystemMessage(content="AI Recruiting Matching Agent initialized.")],
            "job_description": job_description,
            "requirements": {},
            "candidate_pool": [],
            "shortlist": [],
            "ranking_reasoning": "",
            "round_number": round_number,
            "final_report": "",
            "human_feedback": None,
            "awaiting_feedback": False,
            "refinement_history": [],
        }
        result = self.graph.invoke(initial_state, self.config)
        return result

    def send_feedback(self, feedback: str) -> Dict[str, Any]:
        """Continue the graph with human feedback (natural language query)."""
        self.graph.update_state(self.config, {"human_feedback": feedback, "awaiting_feedback": False})
        
        state = self.graph.get_state(self.config)
        result = self.graph.invoke(state.values, self.config)
        return result

    def run_next_round(self) -> Dict[str, Any]:
        """Advance to next screening round (2: deep analysis, 3: final recommendation)."""
        return self.send_feedback("move to next round")

    def get_final_report(self) -> str:
        state = self.graph.get_state(self.config)
        return state.values.get("final_report", "")
    
    







