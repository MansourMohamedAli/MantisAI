import asyncio
import os

from hierarchical_indices_embedding import encode_pdf_hierarchical
from get_embedding_function import get_embedding_function

from helper_functions import *

PATH = "data/Understanding_Climate_Change.pdf"

def retrieve_hierarchical(query, summary_vectorstore, detailed_vectorstore, k_summaries=5, k_chunks=1):
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

async def main(model):
    if os.path.exists("vector_stores/summary_store") and os.path.exists("vector_stores/detailed_store"):
        embeddings = get_embedding_function(model)
        summary_store = FAISS.load_local("vector_stores/summary_store", embeddings, allow_dangerous_deserialization=True)
        detailed_store = FAISS.load_local("vector_stores/detailed_store", embeddings, allow_dangerous_deserialization=True)

    else:
        summary_store, detailed_store = await encode_pdf_hierarchical(path=PATH, model=model)
        summary_store.save_local("vector_stores/summary_store")
        detailed_store.save_local("vector_stores/detailed_store")

    query = "What Life Lifelong learning initiatives are being proviced?"
    results = retrieve_hierarchical(query, summary_store, detailed_store)

    # Print results
    for chunk in results:
        print(f"Page: {chunk.metadata['page']}")
        print(f"Content: {chunk.page_content}...")  # Print first 100 characters
        print("---")



if __name__ == '__main__':
    asyncio.run(main("granite3-dense:8b"))