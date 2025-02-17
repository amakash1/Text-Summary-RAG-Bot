import numpy as np
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "floorgangster"

EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")

class Neo4jVectorStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def store_document(self, doc_id, text):
        """ Store document text along with embeddings and create relationships """
        embedding = EMBEDDING_MODEL.embed_query(text)

        with self.driver.session() as session:
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.text = $text, d.embedding = $embedding
                """,
                doc_id=doc_id, text=text, embedding=embedding
            )

            self.create_relationships(doc_id, embedding)

    def create_relationships(self, doc_id, embedding):
        """ Find similar documents and create SIMILAR_TO relationships """
        with self.driver.session() as session:
            result = session.run(
                "MATCH (d:Document) RETURN d.id AS id, d.embedding AS embedding"
            )

            for record in result:
                existing_id = record["id"]
                existing_embedding = np.array(record["embedding"])

                similarity = np.dot(embedding, existing_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(existing_embedding))

                if similarity > 0.7 and existing_id != doc_id:
                    session.run(
                        """
                        MATCH (a:Document {id: $doc_id}), (b:Document {id: $existing_id})
                        MERGE (a)-[:SIMILAR_TO {score: $similarity}]->(b)
                        """,
                        doc_id=doc_id, existing_id=existing_id, similarity=similarity
                    )

    def search_similar_documents(self, user_query, top_k=3):
        """ Fetch similar documents based on relationships """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document)-[s:SIMILAR_TO]->(other)
                WHERE d.text CONTAINS $user_query
                RETURN other.text AS text, s.score AS similarity
                ORDER BY s.score DESC
                LIMIT $top_k
                """,
                user_query=user_query, 
                top_k=top_k
            )

            return [record["text"] for record in result]


neo4j_store = Neo4jVectorStore()
