import numpy as np
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings

# Neo4j Connection Details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = ""

EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")

class Neo4jVectorStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def store_document(self, doc_id, text):
        """ Store document text along with embeddings """
        embedding = EMBEDDING_MODEL.embed_query(text) # Ensure embedding is stored as list

        with self.driver.session() as session:
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.text = $text, d.embedding = $embedding
                """,
                doc_id=doc_id, text=text, embedding=embedding
            )

    def search_similar_documents(self, query, top_k=3):
        """ Fetch all stored embeddings and compute cosine similarity in Python """
        query_embedding = np.array(EMBEDDING_MODEL.embed_query(query))

        with self.driver.session() as session:
            result = session.run(
                "MATCH (d:Document) RETURN d.text AS text, d.embedding AS embedding"
            )

            docs = []
            for record in result:
                text = record["text"]
                embedding = np.array(record["embedding"])

                # Compute cosine similarity manually
                similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                docs.append((text, similarity))

            # Sort results by similarity score
            docs.sort(key=lambda x: x[1], reverse=True)
            return [doc[0] for doc in docs[:top_k]]

# Initialize Neo4j Vector Store
neo4j_store = Neo4jVectorStore()
