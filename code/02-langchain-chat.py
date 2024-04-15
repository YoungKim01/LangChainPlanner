# %%
import os
from dotenv import load_dotenv
load_dotenv()
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_key = os.environ["AZURE_OPENAI_KEY"]
search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
search_key = os.environ["AZURE_SEARCH_ADMIN_KEY"]

# %%
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

from langchain.vectorstores.azuresearch import AzureSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# %% [markdown]
# ## setup your docs

# %%
from langchain_community.document_loaders import TextLoader

# loader = DirectoryLoader(r"C:\Users\skim333\OneDrive - KPMG\Documents\GitHub\azure-search-vector-samples\demo-python\data\documents_board", glob="**/*.txt")
# documents = loader.load()

# # %%
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
# nodes = text_splitter.split_documents(documents)

# # %%


# # %%
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    api_key=azure_openai_key,
    azure_endpoint=azure_openai_endpoint,
    api_version="2023-09-01-preview",
    chunk_size=1 
)
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=search_endpoint,
    azure_search_key=search_key,
    index_name="boardai03",
    embedding_function=embeddings.embed_query,
)
# vector_store.add_documents(documents=nodes)
# print("done")

# %%
# query = "how do I create a board paper"
# docs = vector_store.similarity_search(
#     query=query,
#     k = 2,
#     search_type = "similarity"
# )

# # %%
# docs

# %%
llm = AzureChatOpenAI(
    deployment_name="gpt-4",
    api_key=azure_openai_key,
    azure_endpoint=azure_openai_endpoint,
    api_version="2023-09-01-preview"
)

# %%
retriever = vector_store.as_retriever(search_key="similarity", search_kwargs={"k": 2})
prompt = hub.pull("rlm/rag-prompt")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
{"context": retriever | format_docs, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
)

# %%
while True:
    query = input("User: ")
    if query == "exit":
        break
    print(rag_chain.invoke(query))




#res = rag_chain.invoke(query)
#res

# %%
#import textwrap
#print("\n".join(textwrap.wrap(res, width = 140)))


