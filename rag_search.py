import os
import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
from lancedb.rerankers import LinearCombinationReranker
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def clip_text(text: str, threshold: int = 350) -> str:
    """Clip text to a maximum length, appending '...' if truncated."""
    return text if len(text) <= threshold else text[:threshold] + "..."

def create_qa_chain(retriever, llm):
    """Create a QA chain with the given retriever and language model."""
    # Define the prompt template
    prompt = PromptTemplate(
        template=(
            "Context information is below.\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query.\n"
            "Query: {input}\n"
            "Answer: "
        ),
        input_variables=["context", "input"]
    )
    
    # Create the document chain and retrieval chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

def setup_vector_store(db):
    """Set up the vector store with embeddings and reranker."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    reranker = LinearCombinationReranker(weight=0.3)
    return LanceDB(
        connection=db,
        embedding=embeddings,
        reranker=reranker,
        table_name="docling"
    )

def print_source_documents(documents):
    """Print the source documents with their metadata."""
    print("\nSource Documents:")
    for i, doc in enumerate(documents, 1):
        print(f"\nSource {i}:")
        metadata = doc.metadata
        print("Filename:", metadata.get("filename", "N/A"))
        if metadata.get("page_numbers"):
            print("Page Numbers:", metadata.get("page_numbers"))
        if metadata.get("title"):
            print("Title:", metadata.get("title"))
        preview = clip_text(doc.page_content, threshold=300)
        print("Content preview:", preview)

def main():
    # Disable parallelism warning in tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Connect to the LanceDB database
    db = lancedb.connect("data/lancedb")
    table = db.open_table("docling")
    
    # Set up the vector store and retriever
    vector_store = setup_vector_store(db)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Initialize the language model
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # Create the QA chain
    rag_chain = create_qa_chain(retriever, llm)
    
    # Define and process the query
    query = "What are the key principles of GDPR data protection?"
    print("Performing RAG query:", query)
    
    # Use the chain with the correct input key
    result = rag_chain.invoke({"input": query})
    
    # Display results
    clipped_answer = clip_text(result["answer"], threshold=350)
    print("\nAnswer:", clipped_answer)
    
    # Display source documents
    print_source_documents(result["context"])

if __name__ == "__main__":
    main()
