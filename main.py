
import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.schema import SystemMessage

from typing import Type
from bs4 import BeautifulSoup
import requests
import json

import streamlit as st


# 加载必要的参数
load_dotenv()
serper_api_key=os.getenv("SERPER_API_KEY")
browserless_api_key=os.getenv("BROWSERLESS_API_KEY")
openai_api_key=os.getenv("OPENAI_API_KEY")

#调用 Google search by Serper
def search(query):
    serper_google_url = os.getenv("SERPER_GOOGLE_URL")
    print(f"Serper Google Search URL: {serper_google_url}")

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", serper_google_url, headers=headers, data=payload)

    print(f'Google 搜索结果: \n {response.text}')
    return response.text


# 根据 url 爬取网页内容，给出最终解答
# target ：分配给 agent 的初始任务
# url ： Agent 在完成以上目标时所需要的URL，完全由Agent自主决定并且选取，其内容或是中间步骤需要，或是最终解答需要
def scrape_website(target: str, url: str):
    print(f"开始爬取： {url}...")

    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    payload = json.dumps({
        "url": url
    })

    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=payload)

    #如果返回成功
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("爬取的具体内容:", text)

        #控制返回内容长度，如果内容太长就需要切片分别总结处理
        if len(text) > 5000:
            #总结爬取的返回内容
            output = summary(target, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

# 如果需要处理的内容过长，先切片分别处理，再综合总结
# 使用 Map-Reduce 方式
def summary(target, content):
    #model list ： https://platform.openai.com/docs/models
    # gpt-4-32k   gpt-3.5-turbo-16k-0613
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    #定义大文本切割器
    # chunk_overlap 是一个在使用 OpenAI 的 GPT-3 或 GPT-4 API 时可能会遇到的参数，特别是当你需要处理长文本时。
    # 该参数用于控制文本块（chunks）之间的重叠量。
    # 上下文维护：重叠确保模型在处理后续块时有足够的上下文信息。
    # 连贯性：它有助于生成更连贯和一致的输出，因为模型可以“记住”前一个块的部分内容。
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=200)


    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {target}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "target"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, target=target)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    target: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, target: str, url: str):
        return scrape_website(target, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")

#初始化 agent 可使用的工具集合
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

#初始话角色详细描述
system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research

            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 5 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)
#初始化 agent 角色模版
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

#初始化大语言模型  负责决策
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

#初始化记忆类型
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=300)

#初始化 agent
agent = initialize_agent(
    tools, #配置工具集
    llm,   #配置大语言模型 负责决策
    agent=AgentType.OPENAI_FUNCTIONS, #设置 agent 类型 https://python.langchain.com/docs/modules/agents/agent_types/openai_functions_agent
    verbose=True,
    agent_kwargs=agent_kwargs, #设定 agent 角色
    memory=memory, #配置记忆模式
)

def main():
    st.set_page_config(page_title="AI Assistant Agent", page_icon=":dolphin:")

    st.header("LangChain 实例讲解 3 -- Agent", divider='rainbow')
    st.header("AI Agent :blue[助理] :dolphin:")

    query = st.text_input("请提问题和需求：")

    if query:
         st.write(f"开始收集和总结资料 【 {query}】 请稍等")

         result = agent({"input": query})

         st.info(result['output'])


def print_hi(message):
    print('===============================================================')
    print(f'{message}')
    #print(f'OpenAI key: {openai_api_key}')
    #print(f'Serper key: {serper_api_key}')
    #print(f'Browserless key: {browserless_api_key}')
    print('===============================================================')

if __name__ == '__main__':
    print_hi('AI Agent 助手 -- LangChain 实例讲解 3')
    main()


