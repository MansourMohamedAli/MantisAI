import asyncio
import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.summarize.chain import load_summarize_chain
from langchain.docstore.document import Document
from hierarchical_indices_embedding import encode_pdf_hierarchical

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
# from evaluation.evalute_rag import *
from helper_functions import encode_pdf, encode_from_string

# Load environment variables from a .env file
load_dotenv()
path = "data/Understanding_Climate_Change.pdf"

# Set the OpenAI API key environment variable
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')



########################################################################################
#                                   Retrieval
########################################################################################

def retrieve_hierarchical(query, summary_vectorstore, detailed_vectorstore, k_summaries=3, k_chunks=5):
    """
    Performs a hierarchical retrieval using the query.

    Args:
        query: The search query.
        summary_vectorstore: The vector store containing document summaries.
        detailed_vectorstore: The vector store containing detailed chunks.
        k_summaries: The number of top summaries to retrieve.
        k_chunks: The number of detailed chunks to retrieve per summary.

    Returns:
        A list of relevant detailed chunks.
    """
    
    # Retrieve top summaries
    top_summaries = summary_vectorstore.similarity_search(query, k=k_summaries)
    
    relevant_chunks = []
    for summary in top_summaries:
        # For each summary, retrieve relevant detailed chunks
        page_number = summary.metadata["page"]
        page_filter = lambda metadata: metadata["page"] == page_number
        page_chunks = detailed_vectorstore.similarity_search(
            query, 
            k=k_chunks, 
            filter=page_filter
        )
        relevant_chunks.extend(page_chunks)
    
    return relevant_chunks



########################################################################################
#                 USER
########################################################################################

async def main():
    if os.path.exists("vector_stores/summary_store") and os.path.exists("vector_stores/detailed_store"):
        embeddings = OpenAIEmbeddings() 
        summary_store = FAISS.load_local("vector_stores/summary_store", embeddings, allow_dangerous_deserialization=True)
        detailed_store = FAISS.load_local("vector_stores/detailed_store", embeddings, allow_dangerous_deserialization=True)

    else:
        summary_store, detailed_store = await encode_pdf_hierarchical(path)
        summary_store.save_local("vector_stores/summary_store")
        detailed_store.save_local("vector_stores/detailed_store")

    query = "What is the greenhouse effect?"
    results = retrieve_hierarchical(query, summary_store, detailed_store)

    # Print results
    for chunk in results:
        print(f"Page: {chunk.metadata['page']}")
        print(f"Content: {chunk.page_content}...")  # Print first 100 characters
        print("---")



if __name__ == '__main__':
    asyncio.run(main())