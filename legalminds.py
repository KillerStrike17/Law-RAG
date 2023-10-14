import os
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import warnings
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage
from chainlit.playground.providers.openai import ChatOpenAI  # importing ChatOpenAI tools
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, DeepLake
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents import load_tools
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.document_loaders.dataframe import DataFrameLoader

warnings.filterwarnings("ignore")
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

review_df = pd.read_csv("./data/justice.csv")

# data = review_df

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 7000, # the character length of the chunk
    chunk_overlap = 700, # the character length of the overlap between chunks
    length_function = len, # the length function - in this case, character length (aka the python len() fn.)
)

# loader = DataFrameLoader(review_df, page_content_column="facts")
# base_docs = loader.load()
# docs = text_splitter.split_documents(base_docs)
# embedder = OpenAIEmbeddings()
# my_activeloop_org_id = "megatron17"
# my_activeloop_dataset_name = "legal_MINDS"
# dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
# vectorstore = DeepLake(dataset_path=dataset_path, embedding=embedder, overwrite = True)
# # vectorstore = DeepLake(dataset_path= embedding=embedder, overwrite=True)
# vectorstore.add_documents(docs)



@cl.on_chat_start  # marks a function that will be executed at the start of a user session
def start_chat():
    
    embedder = OpenAIEmbeddings()

    # This is needed for both the memory and the prompt
    memory_key = "history"
    # Embed and persist db
    persist_directory = "./data/chroma"

    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedder)
    # vectorstore = DeepLake(dataset_path="./legalminds/", embedding=embedder, overwrite=True)
    # vectorstore.add_documents(docs)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    primary_qa_llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k", 
        temperature=0,
    )

    retriever = vectorstore.as_retriever()
    CUSTOM_TOOL_N_DOCS = 3 # number of retrieved docs from deep lake to consider
    CUSTOM_TOOL_DOCS_SEPARATOR ="\n\n" # how to join together the retrieved docs to form a single string
    
    def retrieve_n_docs_tool(query: str) -> str:
        """ Searches for relevant documents that may contain the answer to the query."""
        docs = retriever.get_relevant_documents(query)[:CUSTOM_TOOL_N_DOCS]
        texts = [doc.page_content for doc in docs]
        texts_merged = CUSTOM_TOOL_DOCS_SEPARATOR.join(texts)
        return texts_merged
    
    serp_tool = load_tools(["serpapi"])
    # print("Serp Tool:",serp_tool[0])

    data_tool = create_retriever_tool(
        retriever, 
        "retrieve_n_docs_tool",
        "Searches and returns documents regarding the query asked."
    )
    tools = [data_tool, serp_tool[0]]

    # llm = OpenAIChat(model="gpt-3.5-turbo", temperature=0)
    llm = ChatOpenAI(temperature = 0)


    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)
    system_message = SystemMessage(
            content=(
                "Do your best to answer the questions. "
                "Feel free to use any tools available to look up "
                "relevant information, only if necessary"
            )
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
        )
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
                                   return_intermediate_steps=True)
    # agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)
    # qa_with_sources_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=retriever,
    #     callbacks=[handler],
    #     return_source_documents=True
    # )
    # await msg.update()
    cl.user_session.set("agent", agent_executor)

@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: str):
    agent_executor = cl.user_session.get("agent")

    # result = await qa_with_sources_chain.acall({"query" : message})  #, callbacks=[cl.AsyncLangchainCallbackHandler()])
    # result = agent_executor({"input": message})
    # print("result Dict:",result)
    result = await cl.make_async(agent_executor)(
        {"input": message}, callbacks=[cl.LangchainCallbackHandler()]
    )
    msg =  cl.Message(content=f'The results are:\n{result["output"]}')
    print("message:",msg)
    print("output message:",msg.content)
    # Update the prompt object with the completion
    # msg.content = result["output"]
    # prompt.completion = msg.content
    # msg.prompt = prompt
    # print("message_content: ",msg.content)
    
    msg.send()
    # msg.update()
    return agent_executor
