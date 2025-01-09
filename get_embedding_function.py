# from langchain_community.embeddings.ollama import OllamaEmbeddings # deprecated
from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    # embeddings = OllamaEmbeddings(model="llama3.2",
    #                               base_url='http://127.0.0.1:11434',
    #                               show_progress=True,
    #                               num_ctx=20000,
    #                               num_thread=1)
    embeddings = OllamaEmbeddings(model="llama3.2",
                                  base_url='http://127.0.0.1:11434')
    return embeddings
    
if __name__ == "__main__":
    print(get_embedding_function())
