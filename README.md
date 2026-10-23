# SecondBrain AI

**A local-first, open-source personal knowledge assistant with agentic AI capabilities.**

SecondBrain AI is designed to help people make better use of their own documents and notes. Users can import personal knowledge, ask questions and receive answers grounded in relevant source passages, create notes and tasks, identify learning gaps, and generate study plans.

The project is intended to run on the user's own computer and be shared through GitHub. It is not designed to require cloud deployment or a hosted LLM API.

> **Project status:** This README describes the proposed architecture and implementation approach. Some features are still planned; only describe them as complete after they have been implemented and tested.

## Contents

- [Project Scope](#project-scope)
- [Goals](#goals)
- [Out of Scope](#out-of-scope)
- [Core Features](#core-features)
- [Technology Stack](#technology-stack)
- [System Design](#system-design)
- [Methods and How to Build It](#methods-and-how-to-build-it)
- [Agentic AI Workflow](#agentic-ai-workflow)
- [Suggested Repository Structure](#suggested-repository-structure)
- [Local Setup](#local-setup)
- [Privacy and Security](#privacy-and-security)
- [Testing and Evaluation](#testing-and-evaluation)
- [Current Development Status](#current-development-status)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## Project Scope

### In scope

1. **Local document library**
   - Import PDF and Markdown documents.
   - Extract text, split it into chunks, and index it for semantic search.
   - Keep user documents and application data on the user's computer.

2. **Question answering with Retrieval-Augmented Generation (RAG)**
   - Accept natural-language questions about a user's document collection.
   - Retrieve relevant passages before generating an answer.
   - Include source references so users can inspect the evidence behind an answer.
   - Make it clear when the local knowledge base does not contain enough information.

3. **Persistent notes and tasks**
   - Create and save notes.
   - Create and save tasks.
   - Store structured records in SQLite so they remain available after restarting the app.

4. **Controlled agentic workflows**
   - Use one agent workflow to break a supported request into steps.
   - Select from a small, explicitly permitted set of tools.
   - Inspect tool results and verify that requested changes were actually saved.
   - Ask for confirmation before destructive or consequential actions.

5. **Learning support**
   - Analyze relevant material from the user's documents.
   - Identify potential knowledge gaps based on the available material and user request.
   - Produce a study plan and optionally save its tasks.

6. **Local-first distribution**
   - Publish source code and setup instructions on GitHub.
   - Run inference locally through Ollama after the model has been downloaded.
   - Avoid mandatory cloud AI services and cloud deployment.

### Out of scope for the initial version

- A hosted SaaS service or mandatory cloud deployment.
- Mandatory use of paid cloud LLM APIs.
- Uploading personal documents to GitHub or a remote AI service.
- Multiple collaborating agents, microservices, or complex distributed infrastructure.
- Unrestricted computer control or arbitrary shell-command execution by the agent.
- Claims that generated answers, study plans, or inferred knowledge gaps are guaranteed to be correct.

## Goals

- Make personal information easier to search and reuse.
- Ground AI answers in the user's own documents and show sources.
- Combine document retrieval with practical actions such as saving notes and tasks.
- Demonstrate agentic AI through a bounded, verifiable multi-step workflow.
- Keep setup and operation practical for a personal computer with limited memory.
- Provide a repository that other developers can install, understand, test, and extend.

## Core Features

| Feature | Purpose | Main components |
|---|---|---|
| Document import | Load supported local files | FastAPI, document parsers |
| Semantic search | Find passages relevant to a query | Sentence Transformers, ChromaDB |
| Cited answers | Answer using retrieved passages | Ollama, RAG pipeline |
| Notes | Save and retrieve user notes | FastAPI, SQLite |
| Tasks | Save and retrieve actionable tasks | FastAPI, SQLite |
| Agent workflow | Plan and execute approved multi-step actions | LangGraph and validated tools |
| Study planning | Turn relevant material into a proposed learning plan | RAG plus the agent workflow |
| Local UI | Provide a user-friendly interface | React, Tailwind CSS |

## Technology Stack

The following is the proposed stack; versions and compatibility should be pinned and tested during implementation.

- **Frontend:** React + Tailwind CSS
- **Backend/API:** Python + FastAPI
- **Local LLM runtime:** Ollama
- **Initial model candidate:** Qwen2.5 3B Instruct (`qwen2.5:3b`)
- **Alternative to evaluate:** Gemma 3 4B (`gemma3:4b`)
- **Agent orchestration:** LangGraph
- **Embeddings:** Sentence Transformers, `all-MiniLM-L6-v2`
- **Vector database:** ChromaDB
- **Structured storage:** SQLite
- **Automated tests:** Pytest
- **Source control/distribution:** Git and GitHub

The initial hardware target is a computer with **8 GB RAM**. A small local model, a modest context window (initially around 2,048 tokens), retrieval of roughly 3–5 chunks per question, and one LLM request at a time are sensible starting points—not guaranteed performance figures. Measure memory use, latency, answer quality, and citation accuracy on the target machine.

## System Design

```text
                    ┌──────────────────────────┐
                    │ React + Tailwind UI      │
                    │ Chat, documents, notes,  │
                    │ tasks, study plans       │
                    └────────────┬─────────────┘
                                 │ HTTP / JSON
                    ┌────────────▼─────────────┐
                    │ FastAPI backend          │
                    │ API routes and validation│
                    └──────┬────────────┬──────┘
                           │            │
                 ┌─────────▼─────┐  ┌───▼────────────────┐
                 │ RAG pipeline  │  │ LangGraph workflow │
                 │ parse, chunk, │  │ plan, call approved│
                 │ embed, retrieve│ │ tools, verify      │
                 └──────┬────────┘  └───┬────────────────┘
                        │               │
               ┌────────▼────────┐  ┌───▼────────────────┐
               │ Sentence        │  │ Validated tools     │
               │ Transformers +  │  │ document search,    │
               │ ChromaDB        │  │ create note/task    │
               └────────┬────────┘  └───┬────────────────┘
                        │               │
                        │        ┌──────▼───────┐
                        │        │ SQLite       │
                        │        │ notes/tasks  │
                        │        └──────────────┘
                        │
                 ┌──────▼────────┐
                 │ Ollama local  │
                 │ language model│
                 └───────────────┘
```

**Important design choice:** ordinary questions about documents should go through the RAG pipeline directly. Use the agent only when a request requires planning or multiple actions. This avoids adding agent overhead to simple questions.

## Methods and How to Build It

Build the application incrementally, validating each layer before adding the next.

### 1. Establish the development environment

- Create a Git repository and a Python virtual environment (or use the existing `secondbrain` Conda environment).
- Confirm Python, Node.js, npm, Git, and Ollama are installed.
- Create a minimal FastAPI app and verify its health endpoint.
- Keep dependencies in `requirements.txt` and the frontend's package manifest.
- Add a `.gitignore` before importing personal documents or creating local databases.

**Acceptance check:** the backend starts locally and its health endpoint returns a successful response.

### 2. Verify the local LLM independently

- Install and start Ollama.
- Download the selected model using Ollama.
- Test a simple prompt in the terminal before integrating the model with the application.
- Add a small `backend/llm.py` module to centralize model calls.
- Handle unavailable models, timeouts, and runtime errors with clear messages.

**Acceptance check:** a local prompt produces a response reliably on the target computer.

**Hardware note:** if GPU initialization fails, diagnose the Ollama/runtime issue separately and use a CPU-only run as a comparison test where supported. Do not assume the GPU is the root cause until the diagnostic result supports it.

### 3. Implement document ingestion

For each supported document:

1. Validate the file type and size.
2. Extract text from the PDF or Markdown file.
3. Normalize whitespace and retain useful metadata, such as filename and page number where available.
4. Split the text into overlapping chunks that fit the embedding and LLM context limits.
5. Assign stable document/chunk identifiers and preserve source metadata.
6. Report files that are empty, unreadable, or unsupported instead of silently ignoring them.

Keep document parsing and chunking in a dedicated module, such as `backend/ingestion.py`.

**Acceptance check:** sample documents are imported, chunked, and linked to their source metadata.

### 4. Add embeddings and vector search

- Use `all-MiniLM-L6-v2` through Sentence Transformers to embed chunks and queries.
- Store chunk text, embeddings, and metadata in persistent ChromaDB storage.
- Embed incoming questions and retrieve the most relevant chunks.
- Make the number of retrieved chunks configurable; begin with around 3–5.
- Avoid re-indexing unchanged documents where practical.
- Define how re-importing or deleting a document updates its indexed chunks.

**Acceptance check:** known test questions retrieve passages that actually support their answers, and the index survives an application restart.

### 5. Build RAG answers with citations

1. Receive the user's question.
2. Retrieve relevant document chunks.
3. Construct a prompt containing the question and retrieved evidence.
4. Ask the local LLM to answer using that evidence.
5. Return the answer with source metadata, such as filename and page number when available.
6. If evidence is weak or absent, say so rather than inventing a source-backed answer.

Keep retrieval and answer generation separate so they can be tested independently.

**Acceptance check:** responses include valid source references and do not cite files or passages that were not retrieved.

### 6. Add SQLite persistence for notes and tasks

- Initialize database tables automatically on first startup.
- Define schemas for notes and tasks, including identifiers, titles, content/descriptions, timestamps, and task status where appropriate.
- Use parameterized database operations.
- Validate input lengths and required fields at the API boundary.
- Implement create, list, update, and delete operations only as needed by the UI.
- Require confirmation for destructive actions.

**Acceptance check:** records can be created, read back, and retained after the app restarts.

### 7. Add the controlled LangGraph agent

Start with one graph and a small tool set:

- `search_documents(query)` — retrieve relevant material from the local knowledge base.
- `create_note(title, content)` — save a note in SQLite.
- `create_task(title, description)` — save a task in SQLite.

The workflow should:

1. Interpret the request and decide whether tools are needed.
2. Create a limited plan for supported tasks.
3. Select only from registered tools.
4. Validate tool arguments before execution.
5. Execute each action and inspect the actual result.
6. Verify saved records by reading them back when appropriate.
7. Summarize completed actions and distinguish successful actions from failures.

Do not expose arbitrary shell execution, unrestricted file access, or unvalidated SQL to the LLM. The model proposes actions; application code enforces permissions and validates outcomes.

**Acceptance check:** an end-to-end test can search documents, draft a study plan, save selected tasks, and confirm those tasks exist in SQLite.

### 8. Build the React interface

Create a small UI around the working API:

- Document import and library view.
- Chat/question interface with source references.
- Notes list and editor.
- Task list and status controls.
- Study-plan view.
- Clear loading, empty, and error states.

Use a small API service layer to call FastAPI. Keep backend logic out of UI components, and configure the local API base URL through environment configuration where appropriate.

**Acceptance check:** a user can complete the main workflow from the browser without manually calling API endpoints.

### 9. Connect and validate the end-to-end workflow

Use this as the main demonstration:

1. Import study notes.
2. Ask the application to summarize the material and identify potential learning gaps.
3. Retrieve supporting passages.
4. Generate a study plan based on the retrieved material and the user's request.
5. Present proposed tasks to the user.
6. Save approved tasks through the agent's validated tool.
7. Read back the saved tasks to verify success.
8. Display the plan and source references.

The system must not claim a task was saved unless the database operation succeeded.

### 10. Package the local application

- Document prerequisite installations and supported operating systems.
- Provide a setup script that checks prerequisites and model availability and explains missing dependencies.
- Initialize SQLite and ChromaDB automatically on first launch.
- Provide a launcher for subsequent starts.
- Explain that downloading the model may require an initial internet connection; inference should run locally after setup.
- Include troubleshooting steps for Ollama, model memory usage, ports, and dependency errors.

Keep setup scripts transparent and avoid executing privileged or destructive commands automatically.

## Agentic AI Workflow

The project demonstrates agentic AI through **goal-directed, multi-step work with controlled tool use**, rather than simply sending every prompt to an LLM.

Example request:

> “Review my study notes, identify topics I should revise, create a study plan, and save the tasks.”

The agent can coordinate retrieval, analysis, planning, and persistence. The RAG component supplies evidence from the user's documents; LangGraph coordinates the steps; registered tools perform the permitted database operations; and verification checks whether those operations succeeded.

The agent should be used for tasks that require multiple steps. A simple question such as “What does this document say about topic X?” should use direct RAG question answering.

## Suggested Repository Structure

```text
secondbrain-ai/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   ├── package.json
│   └── ...
├── backend/
│   ├── main.py          # FastAPI app and API routes
│   ├── llm.py           # Ollama integration
│   ├── ingestion.py     # File parsing and chunking
│   ├── rag.py            # Embeddings, retrieval, cited answers
│   ├── agent.py          # LangGraph workflow and tools
│   ├── database.py       # SQLite operations
│   └── models.py         # Request/response and shared models
├── data/
│   └── documents/        # Local runtime data; do not commit user files
├── tests/
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

This is a suggested target structure. The actual repository may be smaller while the first API and model integration are being developed.

## Local Setup

The exact commands can vary by operating system and dependency versions. The current development environment is Windows, with a Conda environment named `secondbrain` and a FastAPI application already started successfully.

### Backend

From the repository root, activate the existing environment or create one:

```bash
conda activate secondbrain
```

Install backend dependencies once the project's `requirements.txt` is prepared:

```bash
pip install -r requirements.txt
```

Start the API from the repository root, adjusting the module path if the app entry point differs:

```bash
uvicorn backend.main:app --reload
```

When the server is running, the current development setup exposes the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

The `POST /generate` endpoint sends a prompt to the local Ollama model:

```json
{
  "prompt": "Summarize the purpose of a personal knowledge assistant."
}
```

It returns the configured model name and generated text. The defaults are `qwen2.5:3b` and `http://127.0.0.1:11434`; override them with the `OLLAMA_MODEL` and `OLLAMA_URL` environment variables. `OLLAMA_TIMEOUT_SECONDS` controls the request timeout (default: 120 seconds). Ollama must be running locally and the configured model must be available.

### Ollama

Install Ollama using its official installer, then download and test the chosen model:

```bash
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
```

Do not proceed with model integration until a terminal prompt succeeds. If the model name or local runtime changes, update the configuration and documentation accordingly.

### Frontend

After the React application and its `package.json` are in place:

```bash
cd frontend
npm install
npm run dev
```

Use the scripts defined in `frontend/package.json`; the exact development command depends on the chosen React tooling.

## Privacy and Security

- Process documents and run inference locally by default.
- Do not upload personal documents, embeddings, SQLite databases, model files, or checkpoints to GitHub.
- Add local data folders, database files, environment files, model artifacts, and runtime checkpoints to `.gitignore`.
- Commit an `.env.example` containing placeholders only, never secrets.
- Validate uploaded file types and sizes.
- Restrict the agent to explicitly registered tools with validated arguments.
- Require user confirmation for destructive or consequential actions.
- Verify database writes and tool outcomes instead of relying on the model's claims.
- Explain any network access required for initial software or model downloads.

“Local-first” reduces the need to send private content to external services, but it does not automatically guarantee security. Review dependencies, file handling, logs, and any optional network features.

## Testing and Evaluation

Use Pytest for backend tests and add frontend tests as the UI grows.

### Functional tests

- Health endpoint starts and responds.
- Supported PDF and Markdown files are parsed.
- Chunk metadata preserves the source filename and available page information.
- Semantic search returns relevant chunks.
- RAG answers include valid references to retrieved chunks.
- Notes and tasks persist after restart.
- Agent tools reject invalid inputs.
- Failed tool operations are reported accurately.
- The study-planning workflow creates only the records the user approved.

### Quality and performance measurements

- **Retrieval quality:** are the most relevant passages returned for a test question?
- **Citation accuracy:** does each source reference point to evidence that supports the answer?
- **Answer grounding:** does the answer stay within the retrieved evidence when required?
- **Latency:** how long do retrieval and generation take?
- **Resource usage:** what are peak RAM and GPU memory requirements?
- **Reliability:** do errors produce clear messages without corrupting the local data?

Create a small, representative test set of documents and questions. Record actual results rather than presenting expected performance as measured fact.

## Current Development Status

Based on the current project notes:

- [x] Created the project directory and a Conda environment named `secondbrain`.
- [x] Installed initial FastAPI dependencies.
- [x] Created `backend/main.py` with basic root and health endpoints.
- [x] Started the initial FastAPI API successfully.
- [x] Installed Ollama.
- [x] Successfully run the chosen local model locally.
- [x] Connect the working local model to FastAPI through `POST /generate`.
- [ ] Implement PDF and Markdown ingestion.
- [ ] Add embeddings, persistent ChromaDB indexing, and RAG citations.
- [ ] Add SQLite-backed notes and tasks.
- [ ] Implement the controlled LangGraph agent workflow.
- [ ] Build the React + Tailwind interface.
- [ ] Add automated tests and complete setup documentation.

**Status update (2026-10-23):** Ollama is working locally and is integrated through the FastAPI `POST /generate` endpoint. The endpoint uses the local `qwen2.5:3b` model by default, with environment-variable configuration and explicit errors for connection, timeout, and upstream response failures. The earlier Windows `llama-server`/CUDA initialization error is no longer blocking development; its cause and the specific change that resolved it have not been recorded. See the project's progress tracker for details.

Update this checklist only after a feature has been implemented and tested.

## Limitations

- Answer quality depends on the model, retrieved evidence, document extraction, and prompt design.
- Scanned PDFs may require OCR, which is not included in the initial scope unless added later.
- Local inference speed and feasible model size depend on the user's hardware.
- Generated study plans and inferred knowledge gaps should be reviewed by the user.
- Citation support depends on retaining correct source metadata during ingestion and retrieval.
- The first installation may need internet access to obtain software dependencies and model weights.

## Contributing

Contributions are welcome. A useful contribution should include:

1. A clear description of the issue or proposed change.
2. A focused implementation consistent with the local-first design.
3. Tests for new behavior or a reproducible explanation when tests are not practical.
4. Documentation updates where setup, configuration, or user-visible behavior changes.
5. Confirmation that personal documents, databases, secrets, and model artifacts are not included in commits.

## License

Choose and add an open-source license before publishing the repository. Until a license file is included, do not assume that others have permission to reuse, modify, or redistribute the code.
