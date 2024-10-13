from langchain_core.documents import Document
import json
import logging

from biokb.llm import AIRLLM
from biokb.embedding import AIRPubmedSearch
from biokb.prompts import agent_prompt
from biokb.utils import get_logger

logger = get_logger('air-agent', debug=True)
logger.info("Starting AIR agent...")


class AIRAgent:
    def __init__(
        self, 
        llm: AIRLLM,
        docstore: AIRPubmedSearch, 
        logger = logger,
        tools: list[callable] = None,
        history: list[dict] = None, 
        trace: list[dict] = [],
    ):
        self.llm = llm
        self.chat = [{"role": "system", "content": agent_prompt}]
        if history:
            self.chat.append(history)
        self.trace = trace
        self.docstore = docstore
        self.logger = logger or logging.getLogger(__name__)
        self.tools = tools
        self.functions = {
            "search_tool": self.search_db,
        }
    
    
    def invoke(
        self, 
        input_text: dict,
        documents: list[Document] = None
    ):
        self.trace.append(input_text)
        self.chat.append(input_text)
        response = self.llm.get_chat_response(self.chat, self.tools)
        self.logger.info(f"Response: {response}")

        try:
            formatted_resp = json.loads(response)
        except json.JSONDecodeError:
            formatted_resp = {"name": "response", "parameters": {"response": response}}
        if formatted_resp['name'] == "response":
            message = {
                "role": "assistant",
                "content": formatted_resp['parameters']['response'],
            }
            self.chat.append(message)
            self.trace.append(message)
            return message, documents

        elif formatted_resp['name'] in self.functions:
            self.trace.append({"role": "assistant", "tool_call": {"type": "function", "function": formatted_resp}})
            response, documents = self.get_func_resp(**formatted_resp)
            self.chat.append({"role": "assistant", "content": response})
            self.trace.append({"role": "assistant", "content": response})
            return {"role": "assistant", "content": response}, documents
        
        else:
            self.trace.pop()
            self.chat.pop()    
            return {
                "role": "assistant",
                "content": "Sorry, I could not understand the request. Please try again."
            }, None

    def get_func_resp(
        self, 
        **params
    ):  
        return self.functions[params['name']](**params['parameters'])

    def search_db(
        self, 
        **params: str
    ):
        return self.docstore.search(self.llm, params['query'])