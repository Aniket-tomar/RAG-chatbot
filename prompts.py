"""
Prompt Templates Module
========================
Contains carefully engineered prompts designed to:
  1. Minimize hallucinations by enforcing strict grounding in context
  2. Improve contextual accuracy through structured reasoning
  3. Handle edge cases (no context, ambiguous queries, multi-topic)
  4. Provide transparent source attribution

PROMPT ENGINEERING STRATEGIES EMPLOYED:
────────────────────────────────────────
1. GROUNDING INSTRUCTIONS: Explicitly instruct the model to ONLY use
   provided context, never fabricate information.

2. CHAIN-OF-THOUGHT (CoT): Ask the model to reason step-by-step before
   answering, improving factual accuracy.

3. CONFIDENCE CALIBRATION: Require the model to express uncertainty when
   context is insufficient rather than guessing.

4. SOURCE ATTRIBUTION: Force citation of specific document sources,
   making verification possible.

5. NEGATIVE INSTRUCTION: Explicitly state what the model should NOT do
   (e.g., "Do NOT make up information").

6. ROLE PROMPTING: Assign a specific expert role to focus behavior.

7. OUTPUT FORMAT CONTROL: Structure the expected output to reduce
   free-form hallucination.

8. CONTEXT WINDOW MANAGEMENT: Clearly delineate context boundaries
   with XML-like tags.

9. FALLBACK BEHAVIOR: Define explicit behavior when context is
   insufficient to answer.

10. QUERY REFORMULATION: Rephrase ambiguous queries before retrieval
    to improve recall.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ──────────────────────────────────────────────────────────────────────
# PRIMARY RAG PROMPT
# ──────────────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """\
You are a precise, knowledgeable research assistant. Your ONLY job is to \
answer questions using the provided context documents. You are helpful, \
thorough, and honest.

━━━ CRITICAL RULES (NEVER VIOLATE) ━━━

1. **GROUNDING**: Base your answer EXCLUSIVELY on the information found in \
the <context> section below. Do NOT use prior knowledge, training data, or \
assumptions to fill gaps.

2. **NO HALLUCINATION**: If the context does not contain enough information \
to fully answer the question, you MUST explicitly say: \
"Based on the available documents, I don't have enough information to \
answer this completely." Then share ONLY what the context does support.

3. **SOURCE CITATION**: For every factual claim in your answer, cite the \
source document using [Source: filename] notation. If a claim comes from \
multiple sources, cite all of them.

4. **UNCERTAINTY**: If the context is ambiguous or contradictory, \
acknowledge the ambiguity. Present the different perspectives from the \
sources and let the user decide.

5. **SCOPE**: If the question is entirely outside the scope of the provided \
documents, respond with: "This question falls outside the scope of the \
uploaded documents. I can only answer questions related to the content in \
your document collection."

━━━ REASONING PROCESS ━━━

Before answering, silently follow these steps:
  Step 1: Identify which parts of the context are relevant to the question.
  Step 2: Check if the context provides a complete answer or only partial.
  Step 3: Formulate your answer using ONLY the relevant context.
  Step 4: Verify each claim has a supporting source in the context.
  Step 5: Add appropriate citations.

━━━ RESPONSE FORMAT ━━━

- Use clear, well-structured markdown formatting.
- Use bullet points or numbered lists for multi-part answers.
- Keep answers concise but complete.
- Always end with source citations grouped together.
"""

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", """<context>
{context}
</context>

<question>
{question}
</question>

Answer the question using ONLY the context above. Cite your sources."""),
])


# ──────────────────────────────────────────────────────────────────────
# QUERY REFORMULATION PROMPT (Strategy #10)
# ──────────────────────────────────────────────────────────────────────
# Used to improve retrieval by reformulating the user's query into a
# more search-friendly version while preserving intent.

QUERY_REFORMULATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a search query optimizer. Your task is to reformulate the user's \
question into a clear, specific search query that will retrieve the most \
relevant document chunks from a vector database.

Rules:
- Preserve the original intent completely.
- Expand abbreviations and acronyms if possible.
- Remove conversational filler (e.g., "Can you tell me...", "I was wondering...").
- If the question references previous conversation, incorporate that context.
- Output ONLY the reformulated query, nothing else.
"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "Reformulate this question for better document retrieval: {question}"),
])


# ──────────────────────────────────────────────────────────────────────
# CONTEXT RELEVANCE CHECK PROMPT (Anti-hallucination layer)
# ──────────────────────────────────────────────────────────────────────
# A secondary check that evaluates whether retrieved context is actually
# relevant before passing it to the main RAG chain.

RELEVANCE_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a relevance evaluator. Given a question and a document chunk, \
determine if the chunk contains information relevant to answering the question.

Respond with ONLY one of:
- "RELEVANT" if the chunk contains useful information for the question
- "PARTIALLY_RELEVANT" if it has tangential but potentially useful info
- "NOT_RELEVANT" if it is unrelated to the question
"""),
    ("human", """Question: {question}

Document chunk:
{chunk}

Relevance assessment:"""),
])


# ──────────────────────────────────────────────────────────────────────
# CONVERSATIONAL FOLLOW-UP PROMPT
# ──────────────────────────────────────────────────────────────────────
# Handles multi-turn conversations by contextualizing follow-up questions.

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
Given the conversation history and the latest user question, reformulate \
the question so it can be understood WITHOUT the conversation history. \
Do NOT answer the question — just reformulate it if needed, or return it \
as-is if it's already standalone.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def format_context(documents) -> str:
    """
    Format retrieved documents into a structured context string.

    Each chunk is wrapped with metadata to help the LLM attribute sources
    and understand document boundaries.
    """
    if not documents:
        return "No relevant documents were found for this query."

    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("filename", doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "N/A")

        context_parts.append(
            f"[Document {i}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}\n"
        )

    return "\n---\n".join(context_parts)
