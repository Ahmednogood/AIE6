from dotenv import load_dotenv
import os
from openai import AsyncOpenAI, OpenAI
from typing import List
import asyncio

class EmbeddingModel:
    def __init__(self, embeddings_model_name: str = "text-embedding-3-small"):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")

        self.async_client = AsyncOpenAI(api_key=self.openai_api_key)
        self.client = OpenAI(api_key=self.openai_api_key)
        self.embeddings_model_name = embeddings_model_name

    async def async_get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
        batch_size = 100
        batches = [list_of_text[i:i + batch_size] for i in range(0, len(list_of_text), batch_size)]

        async def process_batch(batch):
            embedding_response = await self.async_client.embeddings.create(
                input=batch, model=self.embeddings_model_name
            )
            return [e.embedding for e in embedding_response.data]

        results = await asyncio.gather(*[process_batch(batch) for batch in batches])
        return [embedding for batch_result in results for embedding in batch_result]

    def get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
        embedding_response = self.client.embeddings.create(
            input=list_of_text, model=self.embeddings_model_name
        )
        return [e.embedding for e in embedding_response.data]

    def get_embedding(self, text: str) -> List[float]:
        embedding = self.client.embeddings.create(
            input=text, model=self.embeddings_model_name
        )
        return embedding.data[0].embedding

    async def async_get_embedding(self, text: str) -> List[float]:
        embedding = await self.async_client.embeddings.create(
            input=text, model=self.embeddings_model_name
        )
        return embedding.data[0].embedding
