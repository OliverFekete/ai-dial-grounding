import asyncio
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

# System prompt for LLM
SYSTEM_PROMPT = """
You are a helpful assistant. You will receive a RAG context (retrieved user data relevant to the question) and a user question.
Answer the question using only the information provided in the RAG context. If the context does not contain enough information, say you cannot answer.
Be clear and concise. Do not make up information.
"""

# User prompt template
USER_PROMPT = """
## RAG CONTEXT:
{context}

## USER QUESTION:
{query}
"""

def format_user_document(user: dict[str, Any]) -> str:
    result = "User:\n"
    for k, v in user.items():
        result += f"  {k}: {v}\n"
    return result

class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        client = UserClient(DIAL_URL, API_KEY)
        all_users = await client.get_all_users()
        documents = [Document(page_content=format_user_document(user)) for user in all_users]
        self.vectorstore = await self._create_vectorstore_with_batching(documents)
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
        tasks = [
            FAISS.afrom_documents(batch, self.embeddings)
            for batch in batches
        ]
        stores = await asyncio.gather(*tasks)
        final_vectorstore = stores[0]
        for store in stores[1:]:
            final_vectorstore.merge_from(store)
        return final_vectorstore

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        results = await self.vectorstore.asimilarity_search_with_relevance_scores(query, k=k)
        context_parts = []
        for doc, relevance_score in results:
            if relevance_score >= score:
                context_parts.append(doc.page_content)
                print(f"Score: {relevance_score}\nContent:\n{doc.page_content}")
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]
        response = self.llm_client.invoke(messages)
        return response.content

async def main():
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        api_key=API_KEY.get_secret_value(),
        model="text-embedding-3-small-1",
        dimensions=384
    )
    llm_client = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=API_KEY.get_secret_value(),
        api_version="",
        model="gpt-4o"
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            context = await rag.retrieve_context(user_question)
            augmented_prompt = rag.augment_prompt(user_question, context)
            answer = rag.generate_answer(augmented_prompt)
            print(answer)

asyncio.run(main())