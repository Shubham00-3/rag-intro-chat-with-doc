import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Custom Ollama embedding function for ChromaDB
class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name="nomic-embed-text"):
        self.ollama_embeddings = OllamaEmbeddings(model=model_name)
    
    def __call__(self, texts):
        embeddings = []
        for text in texts:
            embedding = self.ollama_embeddings.embed_query(text)
            embeddings.append(embedding)
        return embeddings

# Initialize the Ollama embedding function
ollama_ef = OllamaEmbeddingFunction(model_name="nomic-embed-text")

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=ollama_ef
)

# Initialize Ollama for text generation
ollama_llm = ChatOllama(model="llama2")

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Load documents from the directory
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")
# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print(f"==== Splitting {doc['id']} into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# Function to generate embeddings using Ollama
def get_ollama_embedding(text):
    # Using the same function we created for ChromaDB
    embeddings = ollama_ef([text])
    print("==== Generating embeddings... ====")
    return embeddings[0]  # Return the first (and only) embedding


# Generate embeddings for the document chunks
for doc in chunked_documents:
    print(f"==== Generating embeddings for {doc['id']}... ====")
    doc["embedding"] = get_ollama_embedding(doc["text"])

# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print(f"==== Inserting chunk {doc['id']} into DB ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )


# Function to query documents
def query_documents(question, n_results=2):
    print(f"==== Querying for: {question} ====")
    results = collection.query(query_texts=question, n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = []
    if results["documents"] and len(results["documents"]) > 0:
        for doc in results["documents"][0]:
            relevant_chunks.append(doc)
    
    print(f"==== Found {len(relevant_chunks)} relevant chunks ====")
    return relevant_chunks


# Function to generate a response using Ollama
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    print("==== Generating response with Ollama ====")
    response = ollama_llm.invoke(prompt)
    
    # Return content directly as a string (Ollama response structure is different)
    return response.content


# Interactive mode
def interactive_mode():
    print("\n==== RAG Chat with Documents - Interactive Mode ====")
    print("Type 'exit' to quit")
    
    while True:
        user_question = input("\nEnter your question: ")
        
        if user_question.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Process the question
        relevant_chunks = query_documents(user_question)
        
        if not relevant_chunks:
            print("No relevant information found.")
            continue
        
        answer = generate_response(user_question, relevant_chunks)
        
        print("\nAnswer:")
        print(answer)


# Example query and response generation (or comment this out and use interactive mode)
# question = "tell me about databricks"
# relevant_chunks = query_documents(question)
# answer = generate_response(question, relevant_chunks)
# print(answer)

# Run in interactive mode instead
if __name__ == "__main__":
    interactive_mode()