import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores.azuresearch import AzureSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import textwrap
from langchain_core.prompts import ChatPromptTemplate


class BasicPlugins:
    """
    def:
    
    Usage:

    Examples:

    """
    def __init__(self):
        load_dotenv()
        self.azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        self.azure_openai_key = os.environ["AZURE_OPENAI_KEY"]
        self.search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
        self.search_key = os.environ["AZURE_SEARCH_ADMIN_KEY"]

        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            api_key=self.azure_openai_key,
            azure_endpoint=self.azure_openai_endpoint,
            api_version="2023-09-01-preview",
            chunk_size=1 
        )
        self.vector_store = AzureSearch(
            azure_search_endpoint=self.search_endpoint,
            azure_search_key=self.search_key,
            index_name="boardai03",
            embedding_function=self.embeddings.embed_query,
        )

        self.llm = AzureChatOpenAI(
            deployment_name="gpt-4",
            api_key=self.azure_openai_key,
            azure_endpoint=self.azure_openai_endpoint,
            api_version="2023-09-01-preview",
        )
        

    def people_db(self, task):
        """
        """
        retriever = self.vector_store.as_retriever(search_key="hybrid", search_kwargs={"k": 5})


        prompt = """
        based on the given question's position or team, give a result all the names, emails of who’s in that position or in the team.

        Example:
        'Find out who are the board members and NEC'

        Board Members:
        -	Mike CEO - mike@mycompany.com - CEO
        -	Olivia Johnson - olivia@mycompany.com - CFO
        -	Alex Rodriguez - alex@mycompany.com - COO
        NEC:
        -	Emily Smith - emily@mycompany.com - CTO
        -	Victoria Chen - victoria@mycompany.com - CMO
        -	Michael Turner - michael@mycompany.com - CIO


        \nTask: {task}
        \nContext: {context}


        \nAnswer:""" 

        prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])

        #task = "who are the board members?"
        task = task

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
        {"context": retriever | format_docs, "task": RunnablePassthrough()}
        | prompt_template
        | self.llm
        | StrOutputParser()
        )

        res = rag_chain.invoke(task)

        print("\n".join(textwrap.wrap(res, width = 140)))

        return res   


    def email_function(self,tool_input):
        """
        Functionality: Worker that sends emails to recipient, checks status of the return of those email and chases.
        Inputs: Input will be a list of recipients and content.
        """

        res = "Email sent to: " + tool_input
        return res
    


    def compile_paper(self, boardpaper_pieces):
        """
        Input: list of string(paragraphs) 
        -> RAG
        -> Write board paper 
        """


        retriever = self.vector_store.as_retriever(search_key="hybrid", search_kwargs={"k": 5})


        prompt = """
        Combine all the given board paper pieces, make complete board paper. Don't change the given contents too much unless there's typo.
        Include contents list and agendas on the first page.
        Refer our previous board paper to pick up style.

        \nTask: {task}
        \nContext: {context}

        \nAnswer:""" 

        prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])


        boardpaper_pieces_str = ' | '.join(boardpaper_pieces)
        task = "compile board paper using this pieces: "+ boardpaper_pieces_str
 

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
        {"context": retriever | format_docs, "task": RunnablePassthrough()}
        | prompt_template
        | self.llm
        | StrOutputParser()
        )

        res = rag_chain.invoke(task)

        print("\n".join(textwrap.wrap(res, width = 140)))

        return res   




    def circulate_paper(self):
        """
        """


        return "Res: circulate_paper"
    
