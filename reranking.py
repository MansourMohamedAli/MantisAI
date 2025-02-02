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
# from evaluation.evalute_rag import *


# Helper Classes and Functions

class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")


def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    prompt_template = PromptTemplate(
        input_variables=["query", "doc"],
        template="""On a scale of 1-10, rate the relevance of the following document to the query. Consider the specific context and intent of the query, not just keyword matches.
        Query: {query}
        Document: {doc}
        Relevance Score:"""
    )

    llm = ChatOpenAI(base_url="http://127.0.0.1:11434/v1", temperature=0, model_name="llama3.2:latest", max_tokens=4000, api_key="Ollama")
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)

    scored_docs = []
    for doc in docs:
        input_data = {"query": query, "doc": doc.page_content}
        score = llm_chain.invoke(input_data).relevance_score
        try:
            score = float(score)
        except ValueError:
            score = 0  # Default score if parsing fails
        scored_docs.append((doc, score))

    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_n]]


class CustomRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, num_docs=2) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=30)
        return rerank_documents(query, initial_docs, top_n=num_docs)


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

    print("Comparison of Retrieval Techniques")
    print("==================================")
    print(f"Query: {query}\n")

    print("Baseline Retrieval Result:")
    baseline_docs = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(baseline_docs):
        print(f"\nDocument {i + 1}:")
        print(doc.page_content)

    print("\nAdvanced Retrieval Result:")
    # custom_retriever = CustomRetriever(vectorstore=vectorstore)
    # advanced_docs = custom_retriever._get_relevant_documents(query)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    retriever = CrossEncoderRetriever(
        vectorstore=vectorstore,
        cross_encoder=cross_encoder,
        k=30,
        rerank_top_k=5
    )
    advanced_docs = retriever._get_relevant_documents(query)
    for i, doc in enumerate(advanced_docs):
        print(f"\nDocument {i + 1}:")
        print(doc.page_content)


# Main class
class RAGPipeline:
    def __init__(self, path: str, model: str):
        self.vectorstore = encode_csv(path, model)
        self.model_name = model
        self.llm = ChatOpenAI(base_url="http://127.0.0.1:11434/v1", temperature=0, model_name=self.model_name, max_tokens=4000, api_key="Ollama")

    def run(self, query: str, retriever_type: str = "reranker"):
        if retriever_type == "reranker":
            retriever = CustomRetriever(vectorstore=self.vectorstore)
        elif retriever_type == "cross_encoder":
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            retriever = CrossEncoderRetriever(
                vectorstore=self.vectorstore,
                cross_encoder=cross_encoder,
                k=10,
                rerank_top_k=5
            )
        else:
            raise ValueError("Unknown retriever type. Use 'reranker' or 'cross_encoder'.")

        # qa_chain = RetrievalQA.from_chain_type(
        #     llm=self.llm,
        #     chain_type="stuff",
        #     retriever=retriever,
        #     return_source_documents=True
        # )

        # result = qa_chain.invoke({"query": query})

        # print(f"\nQuestion: {query}")
        # print(f"Answer: {result['result']}")
        # print("\nRelevant source documents:")
        # for i, doc in enumerate(result["source_documents"]):
        #     print(f"\nDocument {i + 1}:")
        #     print(doc.page_content[:200] + "...")


# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--path", type=str, default="data/mantis.csv", help="Path to the document")
    parser.add_argument("--query", type=str, default='how to fix failing lights?', help="Query to ask")
    parser.add_argument("--model", type=str, default='llama3.2:latest', help="Model Name")
    parser.add_argument("--retriever_type", type=str, default="cross_encoder", choices=["reranker", "cross_encoder"],
                        help="Type of retriever to use")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = RAGPipeline(path=args.path, model=args.model)
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
