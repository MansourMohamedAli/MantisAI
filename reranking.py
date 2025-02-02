import os
import sys
from langchain.docstore.document import Document
from typing import List, Any
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field
from langchain_community.document_loaders.csv_loader import CSVLoader
import argparse

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from helper_functions import *

class CrossEncoderRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")
    cross_encoder: Any = Field(description="Cross-encoder model for reranking")
    k: int = Field(default=5, description="Number of documents to retrieve initially")
    rerank_top_k: int = Field(default=3, description="Number of documents to return after reranking")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.cross_encoder.predict(pairs)
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")

def compare_rag_techniques(query: str, docs: List[Document]) -> None:
    embeddings = get_embedding_function("llama3.2:latest", "http://127.0.0.1:11434")
    vectorstore = FAISS.from_documents(docs, embeddings)

    with open('RerankResults.txt','w') as f:
        f.write("Comparison of Retrieval Techniques\n")
        f.write("==================================\n")
        f.write(f"Query: {query}\n")

        f.write("Baseline Retrieval Result:\n")
        baseline_docs = vectorstore.similarity_search(query, k=2)
        for i, doc in enumerate(baseline_docs):
            f.write(f"\nDocument {i + 1}:")
            f.write(doc.page_content)

        f.write("\nAdvanced Retrieval Result:")
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        retriever = CrossEncoderRetriever(
            vectorstore=vectorstore,
            cross_encoder=cross_encoder,
            k=30,
            rerank_top_k=5
        )
        advanced_docs = retriever._get_relevant_documents(query)
        for i, doc in enumerate(advanced_docs):
            f.write(f"\nDocument {i + 1}:")
            f.write(doc.page_content)


# Main class
class RAGPipeline:
    def __init__(self, path: str, model: str):
        self.vectorstore = encode_csv(path, model)
        self.model_name = model
        self.llm = ChatOpenAI(base_url="http://127.0.0.1:11434/v1", temperature=0, model_name=self.model_name, max_tokens=4000, api_key="Ollama")

    def run(self, query: str, retriever_type: str = "reranker"):
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        retriever = CrossEncoderRetriever(
            vectorstore=self.vectorstore,
            cross_encoder=cross_encoder,
            k=10,
            rerank_top_k=5
        )

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--path", type=str, default="data/mantis.csv", help="Path to the document")
    parser.add_argument("--query", type=str, default='Insight is not working?', help="Query to ask")
    parser.add_argument("--model", type=str, default='llama3.2:latest', help="Model Name")
    parser.add_argument("--retriever_type", type=str, default="cross_encoder", choices=["reranker", "cross_encoder"],
                        help="Type of retriever to use")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # pipeline = RAGPipeline(path=args.path, model=args.model)
    # pipeline.run(query=args.query, retriever_type=args.retriever_type)

    # Demonstrate the reranking comparison
    # Example that demonstrates why we should use reranking
    # chunks = [
    #     "The capital of France is great.",
    #     "The capital of France is huge.",
    #     "The capital of France is beautiful.",
    #     """Have you ever visited Paris? It is a beautiful city where you can eat delicious food and see the Eiffel Tower. 
    #     I really enjoyed all the cities in France, but its capital with the Eiffel Tower is my favorite city.""",
    #     "I really enjoyed my trip to Paris, France. The city is beautiful and the food is delicious. I would love to visit again. Such a great capital city."
    # ]
    # docs = [Document(page_content=sentence) for sentence in chunks]
        # Load PDF documents
    loader = CSVLoader(file_path=args.path,
        csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['DR#', 'Problem Summary', 'Problem Description', 'Notes & Resolution']},
        metadata_columns=['DR#', 'Problem Summary', 'Problem Description', 'Notes & Resolution'],
        content_columns=['Problem Summary', 'Problem Description', 'Notes & Resolution'],
        encoding='utf-8')
    docs = loader.load()

    compare_rag_techniques(query=args.query, docs=docs)
