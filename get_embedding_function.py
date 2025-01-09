from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="llama3:latest",
                                  base_url='http://127.0.0.1:11434',
                                  show_progress=True,
                                  num_ctx=20000,
                                  num_thread=1)
    return embeddings
    
if __name__ == "__main__":
    print(get_embedding_function())