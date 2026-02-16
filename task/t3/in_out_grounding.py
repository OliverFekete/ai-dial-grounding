import asyncio
from typing import Any, Optional, Dict, List

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

# SYSTEM PROMPT for NEE and grouping by hobby
SYSTEM_PROMPT = """
You are a hobby extraction and grouping assistant.
Given a user request and a list of user profiles (id and about_me), extract all hobbies mentioned in the request and group user IDs by hobby.
Return a JSON object where each key is a hobby and the value is a list of user IDs who match that hobby.
If no hobbies are found, return an empty object.
Example:
Request: "I need people who love to go to mountains"
Context: [user id and about_me...]
Response:
{
  "rock climbing": [1, 2],
  "hiking": [3, 4],
  "camping": [5]
}
Only use hobbies explicitly mentioned or clearly implied in the user request and about_me fields.
"""

USER_PROMPT = """
## USER REQUEST:
{query}

## USER CONTEXT:
{context}
"""

# Output parser for LLM response
class HobbyUserIDs(BaseModel):
    __root__: Dict[str, List[int]]

def format_user_document(user: dict[str, Any]) -> Document:
    # Only embed id and about_me
    return Document(
        page_content=f"id: {user['id']}\nabout_me: {user.get('about_me', '')}",
        metadata={"id": user["id"]}
    )

class HobbiesWizard:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.embeddings = embeddings
        self.llm_client = llm_client
        self.vectorstore: Optional[Chroma] = None
        self.user_client = UserClient(DIAL_URL, API_KEY)

    async def __aenter__(self):
        print("🔎 Loading all users and building vectorstore...")
        all_users = await self.user_client.get_all_users()
        documents = [format_user_document(user) for user in all_users]
        self.vectorstore = await Chroma.afrom_documents(documents, self.embeddings)
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def update_vectorstore(self):
        # Get all users and all ids from vectorstore
        all_users = await self.user_client.get_all_users()
        all_user_ids = {user["id"] for user in all_users}
        vector_ids = set(self.vectorstore.get()["ids"])
        # Identify new and deleted users
        new_ids = all_user_ids - vector_ids
        deleted_ids = vector_ids - all_user_ids
        # Add new users
        if new_ids:
            new_users = [user for user in all_users if user["id"] in new_ids]
            new_docs = [format_user_document(user) for user in new_users]
            await self.vectorstore.aadd_documents(new_docs)
        # Delete removed users
        if deleted_ids:
            self.vectorstore.delete(list(deleted_ids))

    async def retrieve_context(self, query: str, k: int = 20) -> List[Document]:
        await self.update_vectorstore()
        docs = await self.vectorstore.asimilarity_search(query, k=k)
        return docs

    def augment_prompt(self, query: str, docs: List[Document]) -> str:
        context = "\n".join(doc.page
        _content for doc in docs)
        return USER_PROMPT.format(query=query, context=context)

    def extract_hobby_user_ids(self, augmented_prompt: str) -> Dict[str, List[int]]:
        parser = PydanticOutputParser(pydantic_object=HobbyUserIDs)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]
        response = self.llm_client.invoke(messages)
        return parser.parse(response.content).__root__

    async def output_grounding(self, hobby_user_ids: Dict[str, List[int]]) -> Dict[str, List[dict]]:
        result = {}
        for hobby, user_ids in hobby_user_ids.items():
            users = []
            for uid in user_ids:
                user = await self.user_client.get_user_by_id(uid)
                if user:
                    users.append(user)
            if users:
                result[hobby] = users
        return result

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

    async with HobbiesWizard(embeddings, llm_client) as wizard:
        print("Query samples:")
        print(" - I need people who love to go to mountains")
        print(" - Find users who enjoy painting and hiking")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            docs = await wizard.retrieve_context(user_question)
            if not docs:
                print("No relevant users found.")
                continue
            augmented_prompt = wizard.augment_prompt(user_question, docs)
            hobby_user_ids = wizard.extract_hobby_user_ids(augmented_prompt)
            if not hobby_user_ids:
                print("{}")
                continue
            result = await wizard.output_grounding(hobby_user_ids)
            print(result)

if __name__ == "__main__":
    asyncio.run(main())