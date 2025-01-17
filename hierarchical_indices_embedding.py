import asyncio
from dotenv import load_dotenv
from langchain.chains.summarize.chain import load_summarize_chain
from langchain.docstore.document import Document
from langchain_ollama import ChatOllama
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from get_embedding_function import get_embedding_function
from helper_functions import *

async def encode_pdf_hierarchical(path, model, base_url, chunk_size=200, chunk_overlap=100):
    """
    Asynchronously encodes a PDF book into a hierarchical vector store using OpenAI embeddings.
    Includes rate limit handling with exponential backoff.
    
    Args:
        path: The path to the csv file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.
        
    Returns:
        A tuple containing two FAISS vector stores:
        1. Document-level summaries
        2. Detailed chunks
    """
    
    # Load CSV
    try:
        loader = CSVLoader(file_path=path,
            csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': ['DR#', 'Problem Summary', 'Problem Description', 'Notes & Resolution']},
            metadata_columns=['DR#', 'Problem Summary', 'Problem Description', 'Notes & Resolution'],
            content_columns=['Problem Summary', 'Problem Description', 'Notes & Resolution'],
            encoding='utf-8')
    except RuntimeError as e:
        print(e)
        quit()
    documents = await asyncio.to_thread(loader.load)
    
    # Create document-level summaries
    # summary_llm = ChatOllama(base_url=f'{base_url}/v1', temperature=0, model_name="llama3.2", max_tokens=4000)
    summary_llm = ChatOpenAI(base_url=f'{base_url}/v1', temperature=0, model_name="llama3.2", max_tokens=4000, api_key="Ollama")
    summary_chain = load_summarize_chain(summary_llm, chain_type="map_reduce")
    
    async def summarize_doc(doc):
        """
        Summarizes a single document with rate limit handling.
        
        Args:
            doc: The document to be summarized.
            
        Returns:
            A summarized Document object.
        """
        # Retry the summarization with exponential backoff
        # summary_output = await retry_with_exponential_backoff(summary_chain.ainvoke([doc]))
        summary_output = await summary_chain.ainvoke([doc])
        summary = summary_output['output_text']
        print(f'{doc.metadata["DR#"]}: {summary}\n---\n')
        return Document(
            page_content=summary,
            # metadata={"source": path, "row": doc.metadata["row"], "summary": True}
            # metadata={"source": path, "DR#": doc.metadata["DR#"], "Summary#": doc.metadata["Problem Summary"], "summary": True}
            metadata={"source": path, "DR#": doc.metadata["DR#"], "summary": True}
        )

    # Process documents in smaller batches to avoid rate limits
    batch_size = 5  # Adjust this based on your rate limits
    summaries = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_summaries = await asyncio.gather(*[summarize_doc(doc) for doc in batch])
        summaries.extend(batch_summaries)
        await asyncio.sleep(1)  # Short pause between batches


    # Split documents into detailed chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    detailed_chunks = await asyncio.to_thread(text_splitter.split_documents, documents)

    # Update metadata for detailed chunks
    for i, chunk in enumerate(detailed_chunks):
        chunk.metadata.update({
            "chunk_id": i,
            "summary": False,
            "DR#": chunk.metadata.get("DR#", 0)
        })
        # print(chunk.metadata['DR#'])

    # Create embeddings
    # embeddings = OpenAIEmbeddings(base_url='http://localhost:11434/v1/embeddings', api_key='ollama')
    embeddings = get_embedding_function(model, base_url)

    # Create vector stores asynchronously with rate limit handling
    async def create_vectorstore(docs):
        """
        Creates a vector store from a list of documents with rate limit handling.
        
        Args:
            docs: The list of documents to be embedded.
            
        Returns:
            A FAISS vector store containing the embedded documents.
        """
        return await retry_with_exponential_backoff(
            asyncio.to_thread(FAISS.from_documents, docs, embeddings)
        )

    # Generate vector stores for summaries and detailed chunks concurrently
    summary_vectorstore, detailed_vectorstore = await asyncio.gather(
        create_vectorstore(summaries),
        create_vectorstore(detailed_chunks)
    )

    return summary_vectorstore, detailed_vectorstore

if __name__ == '__main__':
    PATH = "mantis.csv"
    asyncio.run(encode_pdf_hierarchical(PATH, 'phi4:latest', 'http://127.0.0.1:11434'))