import asyncio
import os

from hierarchical_indices_embedding import encode_pdf_hierarchical
from get_embedding_function import get_embedding_function
from helper_functions import *

def retrieve_hierarchical(query, summary_vectorstore, detailed_vectorstore, k_summaries=3, k_chunks=10):
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
    # scores = summary_vectorstore.similarity_search_with_relevance_scores(query, k=k_summaries)
    # print(scores)
    
    relevant_chunks = []
    for summary in top_summaries:
        # For each summary, retrieve relevant detailed chunks
        dr_num = summary.metadata["DR#"]
        dr_filter = lambda metadata: metadata["DR#"] == dr_num
        page_chunks = detailed_vectorstore.similarity_search(
            query, 
            k=k_chunks,
            filter=dr_filter
        )
        relevant_chunks.extend(page_chunks)
    return relevant_chunks

async def main(query, model, base_url, path):
    if os.path.exists("vector_stores/summary_store") and os.path.exists("vector_stores/detailed_store"):
        embeddings = get_embedding_function(model, base_url)
        summary_store = FAISS.load_local("vector_stores/summary_store", embeddings, allow_dangerous_deserialization=True)
        detailed_store = FAISS.load_local("vector_stores/detailed_store", embeddings, allow_dangerous_deserialization=True)

    else:
        summary_store, detailed_store = await encode_pdf_hierarchical(path, model, base_url)
        summary_store.save_local("vector_stores/summary_store")
        detailed_store.save_local("vector_stores/detailed_store")
        
    # summary_store.similarity_search_with_relevance_scores
    results = retrieve_hierarchical(query, summary_store, detailed_store)

    # Print results
    for chunk in results:
        print(f"DR#: {chunk.metadata['DR#']}")
        # print(chunk.metadata['Problem Description'])
        # print("---")
        # print(chunk.metadata['Notes & Resolution'])


if __name__ == '__main__':
    PATH = "mantis.csv"
    query = 'My panel graphics are not working.'
    asyncio.run(main(query, 'phi4:latest', 'http://127.0.0.1:11434', PATH))