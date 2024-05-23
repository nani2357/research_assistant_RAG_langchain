import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# Retrieve OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")



## langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

no_of_results =3
ddg_search =   DuckDuckGoSearchAPIWrapper()

def web_search(query:str, num_results:int=no_of_results) :
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]


SUMMARY_TEMPLATE = """{text} 
-----------
Using the above text, answer in short the following question: 
> {question}
-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501



promt = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


def scrape_text(url: str):
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            page_text = soup.get_text(separator=" ",strip=True)
            return page_text
        else:
            return f"Failed to retrive the webpage: Status Code {response.status_code}"
        
    except Exception as e:
        print(e)
        return f"failed to retrive the webpage: {e}"

url = "https://blog.langchain.dev/announcing-langsmith/"

#page_content = scrape_text(url)[:5000]

chain = RunnablePassthrough.assign(
    text = lambda x: scrape_text(x["url"])[:5000]
    ) | promt | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()


chain2 =RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"])
) | (lambda x: [{"question": x["question"], "url":u}for u in x["urls"]]) |  chain.map()


chain2.invoke(
    {
        "question": "What is Langsmith?",
        "url":url
    }
)
