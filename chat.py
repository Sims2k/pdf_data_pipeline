import streamlit as st
import lancedb
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()


# Initialize LanceDB connection
@st.cache_resource
def init_db():
    """Initialize database connection.

    Returns:
        LanceDB table object
    """
    db = lancedb.connect("data/lancedb")
    return db.open_table("docling")


def get_context(query: str, table, num_results: int = 10) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    results = table.search(query).limit(num_results).to_pandas()
    contexts = []

    for _, row in results.iterrows():
        # Extract metadata
        filename = row["metadata"]["filename"]
        page_numbers = row["metadata"]["page_numbers"]
        title = row["metadata"]["title"]

        # Build source citation
        source_parts = []
        if filename:
            source_parts.append(filename)
        if page_numbers is not None and len(page_numbers) > 0:
            source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")

        source = f"\nSource: {' - '.join(source_parts)}"
        if title:
            source += f"\nTitle: {title}"

        contexts.append(f"{row['text']}{source}")

    return "\n\n".join(contexts)


def get_chat_response(messages, context: str) -> str:
    """Get streaming response from OpenAI API.

    Args:
        messages: Chat history
        context: Retrieved context from database

    Returns:
        str: Model's response
    """
    system_prompt = f"""You are a GDPR compliance assistant. Answer questions based on the provided context.
Use only the information from the context to answer questions. If you're unsure or the context
doesn't contain the relevant information, say so.

Context:
{context}
"""

    messages_with_context = [{"role": "system", "content": system_prompt}, *messages]

    try:
        with st.spinner("Generating response..."):
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=messages_with_context,
                temperature=0.7,
                stream=True,
            )
            response = st.write_stream(stream)
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, an error occurred while generating the response."


# Initialize Streamlit app
st.title("🔒 GDPR Compliance Q&A")

# Inject global CSS styling for search results
st.markdown(
    """
    <style>
    .search-result {
        margin: 10px 0;
        padding: 10px;
        border-radius: 4px;
        background-color: #e8f0fe;
    }
    .search-result summary {
        cursor: pointer;
        color: #0a66c2;
        font-weight: 500;
    }
    .search-result summary:hover {
        color: #0056b3;
    }
    .metadata {
        font-size: 0.9em;
        color: #333;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add sidebar instructions
st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    - Ask a GDPR-related question in the input box.
    - The app will retrieve relevant sections from your documents.
    - The assistant will generate a response based solely on the retrieved context.
    - Use the "Clear Conversation" button to reset the chat.
    """
)

# Add "Clear Conversation" button to reset chat history
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.experimental_rerun()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize database connection
table = init_db()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a GDPR-related question"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get relevant context with spinner
    with st.spinner("Searching document..."):
        context = get_context(prompt, table)
    
    # Display found relevant sections
    st.write("Found relevant sections:")
    for chunk in context.split("\n\n"):
        parts = chunk.split("\n")
        if not parts or not parts[0].strip():
            continue  # Skip empty chunks

        text = parts[0].strip()
        metadata = {
            line.split(": ")[0]: line.split(": ")[1]
            for line in parts[1:]
            if ": " in line
        }
        source = metadata.get("Source", "Unknown source")
        title = metadata.get("Title", "Untitled section")

        if text:  # Ensure text is not empty
            st.markdown(
                f"""
                <div class="search-result">
                    <details>
                        <summary>{source}</summary>
                        <div class="metadata">Section: {title}</div>
                        <div style="margin-top: 8px;">{text}</div>
                    </details>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Get model response with streaming
    with st.chat_message("assistant"):
        response = get_chat_response(st.session_state.messages, context)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
