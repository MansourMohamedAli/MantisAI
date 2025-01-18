import asyncio
import os
import json

from hierarchical_indices_embedding import encode_pdf_hierarchical
from get_embedding_function import get_embedding_function
from helper_functions import *

def retrieve_hierarchical(query, summary_vectorstore, detailed_vectorstore, k_summaries=3, k_chunks=50):
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
        dr_num = summary.metadata["DR#"]
        dr_filter = lambda metadata: metadata["DR#"] == dr_num
        page_chunks = detailed_vectorstore.similarity_search(
            query, 
            k=k_chunks,
            filter=dr_filter
        )
        relevant_chunks.extend(page_chunks)
    return relevant_chunks

async def main(query, model, base_url, path, summary_store_path, detailed_store_path, output_path, k_summaries, k_chunks, embedding_chunk_size, embedding_chunk_overlap):
    if os.path.exists(summary_store_path) and os.path.exists(detailed_store_path):
        embeddings = get_embedding_function(model, base_url)
        summary_store = FAISS.load_local(summary_store_path, embeddings, allow_dangerous_deserialization=True)
        detailed_store = FAISS.load_local(detailed_store_path, embeddings, allow_dangerous_deserialization=True)

    else:
        summary_store, detailed_store = await encode_pdf_hierarchical(path, model, base_url, output_path, int(embedding_chunk_size), int(embedding_chunk_overlap))
        summary_store.save_local(summary_store_path)
        detailed_store.save_local(detailed_store_path)
        
    # summary_store.similarity_search_with_relevance_scores
    results = retrieve_hierarchical(query, summary_store, detailed_store, int(k_summaries), int(k_chunks))

    # Print results
    for chunk in results:
        # print(chunk)
        print(f"DR#: {chunk.metadata['DR#']}")
        print(chunk.metadata['Problem Description'])
        print(chunk.metadata['Notes & Resolution'])
        # print("---")


if __name__ == '__main__':
    config_path = "llm_config/config.json"

    with open(config_path) as f:
        config_json = json.load(f)

    config_values = config_json.values()
    print(config_values)
    asyncio.run(main(*config_values))