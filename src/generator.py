"""
generator.py
------------
Responsible for:
  1. Defining the prompt template that injects retrieved chunks + user query
  2. Initializing the Groq LLM (Llama-3.3-70B)
  3. Running the RAG chain (retriever → prompt → LLM)
  4. Streaming the response token-by-token back to the caller

WHY Groq + Llama-3.3-70B?
  - Llama-3.3-70B is an instruction-optimized model (fulfills assignment requirement)
  - Groq provides free-tier API with extremely fast inference (~500 tokens/sec)
  - No GPU needed — cloud API handles compute
  - Groq's Python SDK supports streaming natively through LangChain
  - Alternative (Mistral/Zephyr local) would need 16GB+ RAM — not practical locally
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Load environment variables from .env file (GROQ_API_KEY lives here)
load_dotenv()


# ── Constants ─────────────────────────────────────────────────────────────────

# Groq model identifier
# WHY llama-3.3-70b-versatile?
#   - 70B parameters → strong reasoning over complex legal text
#   - "versatile" variant is optimized for instruction-following tasks
#   - Groq's fastest model for this size class
MODEL_NAME = "llama-3.3-70b-versatile"

# Maximum tokens in the LLM response
# 1024 is enough for detailed answers without running too long
MAX_TOKENS = 1024

# Temperature controls randomness in generation
# WHY 0.1 (near zero)?
#   We want FACTUAL answers grounded in the document — not creative responses.
#   Low temperature = the model sticks close to what the retrieved context says.
#   High temperature = more creative but risks hallucination.
TEMPERATURE = 0.1


# ── Prompt Template ───────────────────────────────────────────────────────────

# WHY this specific prompt structure? (This is the most important thing to explain in review)
#
# 1. SYSTEM message sets the persona and strict rules:
#    - "legal document assistant" → focuses the model on the task
#    - "ONLY use the provided context" → prevents hallucination from training data
#    - "If not found, say so" → forces honest "I don't know" instead of making things up
#
# 2. Context injection with [CHUNK X] labels:
#    - Labels help the model reference which chunk an answer came from
#    - Numbered chunks make it easy to trace answers back to source
#
# 3. Concise answer instruction:
#    - Legal documents can trigger very long responses
#    - We want clear, direct answers that users can actually read

SYSTEM_PROMPT = """You are a helpful legal document assistant specializing in eBay's User Agreement.

Your job is to answer user questions STRICTLY based on the provided document context below.

Rules you MUST follow:
1. ONLY use information from the provided context chunks — do not use your general knowledge
2. If the answer is NOT found in the context, clearly say: "This information is not covered in the provided document."
3. Always be precise and factual — this is a legal document
4. Keep answers concise but complete
5. If relevant, mention which section of the agreement the information comes from

CONTEXT FROM DOCUMENT:
{context}
"""

HUMAN_PROMPT = "Question: {question}"


def format_context(docs: list[Document]) -> str:
    """
    Format retrieved Document chunks into a clean context string for the prompt.

    WHY label each chunk with [CHUNK X]?
      The model can reference chunks explicitly in its answer.
      Also helps during debugging — we can match the model's references
      back to actual chunks to verify grounding.

    Args:
        docs: List of retrieved Document objects from the retriever.

    Returns:
        Formatted string with all chunks labeled and separated.
    """
    formatted_chunks = []

    for i, doc in enumerate(docs):
        chunk_label = f"[CHUNK {i+1}] (Source: {doc.metadata.get('source', 'Unknown')})"
        formatted_chunks.append(f"{chunk_label}\n{doc.page_content}")

    # Separate chunks with a clear divider so the model can distinguish them
    return "\n\n---\n\n".join(formatted_chunks)


def get_llm(streaming: bool = True) -> ChatGroq:
    """
    Initialize the Groq LLM instance.

    WHY streaming=True as default?
      The assignment requires real-time streaming responses.
      When streaming=True, the LLM yields tokens one at a time as they're generated,
      which we then display in Streamlit using st.write_stream().

      Without streaming, the user would see a loading spinner for 5-10 seconds
      then the full response appears at once — bad UX.

    WHY ChatGroq over direct Groq SDK?
      LangChain's ChatGroq integrates seamlessly with prompt templates and chains.
      It handles message formatting (system/human/assistant roles) automatically.

    Args:
        streaming: Whether to enable token-by-token streaming.

    Returns:
        Initialized ChatGroq LLM instance.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Create a .env file with: GROQ_API_KEY=your_key_here"
        )

    llm = ChatGroq(
        model=MODEL_NAME,
        groq_api_key=api_key,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        streaming=streaming,       # Enable token-by-token streaming
    )

    return llm


def build_rag_chain(llm: ChatGroq):
    """
    Build the complete RAG chain using LangChain's LCEL (LangChain Expression Language).

    HOW the chain works (explain in interview):
      Input: {"context": formatted_chunks_string, "question": user_query}
        ↓
      ChatPromptTemplate → formats system + human messages with the input values
        ↓
      ChatGroq LLM → generates response (streams if streaming=True)
        ↓
      StrOutputParser → extracts the text content from the LLM's response object

    WHY LCEL (the pipe | operator)?
      LCEL creates a lazy evaluation chain — each component is a Runnable.
      This means we can call chain.stream() to get an iterator of token chunks,
      which is exactly what Streamlit's st.write_stream() expects.

    WHY NOT use RetrievalQA chain?
      LangChain's RetrievalQA is a higher-level abstraction — it hides the retrieval step.
      We want explicit control over retrieval (our Retriever class handles it) so we can:
        - Show source chunks in the UI
        - Log retrieved chunks for debugging
        - Handle empty retrieval gracefully
      Manual chain gives us that control.

    Args:
        llm: Initialized ChatGroq instance.

    Returns:
        A LangChain Runnable chain that accepts {"context": str, "question": str}.
    """
    # Define the prompt template with system and human message roles
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    # Build the chain using LCEL pipe syntax
    # RunnablePassthrough() passes the input dict unchanged to the prompt
    chain = (
        RunnablePassthrough()   # Takes {"context": ..., "question": ...} as-is
        | prompt                # Formats into ChatPromptValue with system + human messages
        | llm                   # Sends to Groq API, gets AIMessage back (or stream)
        | StrOutputParser()     # Extracts .content string from AIMessage
    )

    return chain


def generate_response(
    query: str,
    docs: list[Document],
    streaming: bool = True,
):
    """
    Generate a response for the user's query using retrieved document chunks.

    This is the main function called by app.py on every user message.

    For streaming (streaming=True):
      Returns a generator that yields string tokens one at a time.
      Streamlit's st.write_stream() consumes this generator.

    For non-streaming (streaming=False):
      Returns the complete response string at once.
      Used for testing and the notebook.

    Args:
        query     : The user's question.
        docs      : Retrieved Document chunks from retriever.py.
        streaming : Whether to stream the response.

    Returns:
        If streaming=True  → generator yielding string tokens
        If streaming=False → complete response string
    """
    # Handle case where no relevant chunks were found
    if not docs:
        no_context_msg = (
            "I couldn't find relevant information in the document to answer your question. "
            "Please try rephrasing or ask about topics covered in the eBay User Agreement."
        )
        if streaming:
            # Yield the message as a single chunk so st.write_stream() works
            def no_context_generator():
                yield no_context_msg
            return no_context_generator()
        return no_context_msg

    # Format retrieved chunks into the context string
    context = format_context(docs)

    # Build the RAG chain with appropriate streaming setting
    llm = get_llm(streaming=streaming)
    chain = build_rag_chain(llm)

    # Prepare the input dict for the chain
    chain_input = {
        "context": context,
        "question": query,
    }

    if streaming:
        # chain.stream() returns an iterator of string tokens
        # We return it directly — Streamlit will consume it
        return chain.stream(chain_input)
    else:
        # chain.invoke() returns the complete response string
        return chain.invoke(chain_input)


# ── Script entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test the generator directly (non-streaming):
    # $ python src/generator.py
    from retriever import Retriever

    retriever = Retriever()

    test_query = "What happens if I have a dispute with eBay?"
    print(f"\nQuery: {test_query}\n")

    docs = retriever.retrieve(test_query)
    response = generate_response(test_query, docs, streaming=False)

    print(f"Response:\n{response}")
