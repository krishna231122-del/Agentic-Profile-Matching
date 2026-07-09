## Agentic-Profile-Matching

An LLM-powered recruiting assistant built with **LangGraph** that automates resume screening — from job description parsing to multi-round candidate ranking, explainable match reports, and interactive natural-language refinement. File system access is exposed via a standards-compliant **MCP (Model Context Protocol) server**.

---

## Features

### Part A — Agent Architecture
- **LangGraph state machine** with persistent conversation memory (`MemorySaver`)
- **Workflow:** `START → Parse JD → Extract Requirements → Search Resumes → Rank Candidates → Generate Report → Human Feedback Loop → END`
- **Agent State** tracks:
  - Full conversation history (`messages`)
  - Parsed job requirements (must-have / nice-to-have)
  - Candidate pool, shortlist, and ranking reasoning at every round
- **Tools:**
  - File system tools (`list_resumes`, `read_resume`, `save_report`, `batch_process`, `watch_directory`) — served via MCP
  - `rag_search` — semantic resume search (Mistral embeddings + Chroma vector DB)
  - `extract_requirements(jd)` — parses must-have vs nice-to-have
  - `compare_candidates(candidate_ids)` — head-to-head comparison
  - `generate_interview_questions(candidate_id)` — tailored screening questions

### Part B — Interactive Features
- Natural language queries handled via intent routing:
  - *"Find me candidates with React and 3+ years experience"*
  - *"Compare the top 3 matches side by side"*
  - *"Why did John rank higher than Jane?"*
- **Iterative refinement** — adjust requirements mid-conversation; agent re-ranks and explains what changed

### Part C — Advanced Capabilities
- **Multi-round screening:**
  1. Initial screen → top N from full candidate pool
  2. Deep analysis → verifies skill depth, experience relevance, career trajectory
  3. Final round → hire / no-hire / borderline recommendation
- **Explainability:**
  - Detailed markdown match reports per candidate
  - Strengths mapped to requirements, gaps clearly flagged
  - Concrete improvement suggestions for borderline candidates

---
##Project Structure
## 📁 Project Structure

```text
your_project/
│
├── main.py                    # Entry point — run this to start the agent
├── matching_agent.py          # LangGraph agent: state, tools, nodes, graph
├── filesystem_mcp_server.py   # MCP server exposing file system tools
│
├── Resume_dir/                # Candidate resumes (.txt, .pdf, .docx)
│
├── chroma_resume_db/          # Auto-created vector DB (gitignored)
│
├── reports/                   # Auto-created saved reports (gitignored)
│
├── .env                       # API keys (gitignored, not committed)
├── .env.example               # Template for required environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.11+
- A [Mistral AI](https://console.mistral.ai/) API key (for both chat and embeddings)

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`
-langgraph
-langchain-core
-langchain-mistralai
-langchain-chroma
-langchain-text-splitters
-chromadb
-mcp
-watchdog
-pypdf
-python-docx
-python-dotenv

