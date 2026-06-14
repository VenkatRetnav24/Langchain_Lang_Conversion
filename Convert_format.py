import os
import anthropic
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

model = ChatAnthropic(model ='claude-sonnet-4-5')
parser = StrOutputParser()


topic_prompt = ChatPromptTemplate.from_template(
    "please write about {topic}"
)

translation_prompt = ChatPromptTemplate.from_template(
    """
        Translate the topic in language Tamil:
        Text:{text}
    """
)

topic_chain = topic_prompt | model | parser

translation_chain = (
    {"text": lambda x: x}
    | translation_prompt
    | model
    | parser   
) 

final_chain = topic_chain | translation_chain

response = final_chain.invoke({
        "topic":"AI"       
    })

prompt = ChatPromptTemplate.from_template(
    """
    Format the response
    {response}
    Query:{topic}
    """
)

#output parser
jsonparser = JsonOutputParser()

final_chain=prompt.partial(
    response = jsonparser.get_format_instructions()
) |model | jsonparser

print(response)



