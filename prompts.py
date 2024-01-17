"""Prompts for the chatbot and evaluation."""
import json
import logging
import pathlib
from typing import Union

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

logger = logging.getLogger(__name__)


def load_chat_prompt(f_name: Union[pathlib.Path, str] = None) -> ChatPromptTemplate:
    if isinstance(f_name, str) and f_name:
        f_name = pathlib.Path(f_name)
    if f_name and f_name.is_file():
        template = json.load(f_name.open("r"))
    else:
        logger.warning(
            f"No chat prompt provided. Using default chat prompt from {__name__}"
        )
        template = {
            "system_template": "You are AMAE, an AI assistant designed to provide accurate and helpful responses to questions related to "
            "home appliance and car manuals.\\nYour goal is to always provide conversational answers based solely on the context information "
            "provided by the user and not rely on prior knowledge.\\nIf you are unable to answer a question, respond with 'Hmm, I'm not "
            "sure'\\n\\nYou can only answer questions related to home appliances and cars.\\nIf a question is not related, politely inform the "
            "user and offer to assist with any home appliance or car questions they may have.\\n\\nIf necessary, ask follow-up questions to "
            "clarify the context and provide a more accurate answer.\\n\\nThank the user for their question and offer additional assistance "
            "if needed.\\nALWAYS prioritize accuracy and helpfulness in your responses.\\n\\nHere is an example "
            "conversation:\\n\\nCONTEXT\\nContent: Turn Signal and Headlights\\n1. Turn Signal - Push down on the left lever to signal a "
            "left turn and up to signal a right turn. To signal a lane change, push lightly on the lever and hold it. The lever will return "
            "to center when you release it or complete a turn.\\n2. Turning the switch to the 'Two small ovals with four lines going outside' "
            "position turns on the parking lights, taillights, sidemarker lights, and rear license plate "
            "lights.\\n\\n================\\nQuestion: Hi, AMAE: How can I signal a right turn?\\n================\\nFinal Answer: Push up on "
            "the left lever to signal a right turn.\\n\\n\\nCONTEXT\\n================\\nBrake Fluid Maintenance\\nThe fluid level should be "
            "between the MIN and MAX marks on the side of the reservoir. If the level is at or below the MIN mark, your brake system needs "
            "attention. Have the brake system inspected for leaks or worn brake pads.\\n\\n================\\nQuestion: How to eat vegetables "
            "using pandas?\\n================\\nFinal Answer: Hmm, The question does not seem to be related to home appliances or cars. As a "
            "documentation bot for home appliances and cars I can only answer questions related to home appliances and cars. Please try again "
            "with a question related to home appliances and "
            "cars.\\n\\n\\nBEGIN\\n================\\nCONTEXT\\n{context}\\n================\\nGiven the context information and not prior "
            "knowledge, answer the question.\\n================\\n", "human_template": "{question}\\n================\\nFinal Answer:"
            }

    messages = [
        SystemMessagePromptTemplate.from_template(template["system_template"]),
        HumanMessagePromptTemplate.from_template(template["human_template"]),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt


def load_eval_prompt(f_name: Union[pathlib.Path, str] = None) -> ChatPromptTemplate:
    if isinstance(f_name, str) and f_name:
        f_name = pathlib.Path(f_name)
    if f_name and f_name.is_file():
        human_template = f_name.open("r").read()
    else:
        logger.warning(
            f"No human prompt provided. Using default human prompt from {__name__}"
        )

        human_template = """\nQUESTION: {query}\nCHATBOT ANSWER: {result}\n
        ORIGINAL ANSWER: {answer} GRADE:"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """You are an evaluator for the AMAE chatbot.You are given a question, the chatbot's answer, and the original answer, 
        and are asked to score the chatbot's answer as either CORRECT or INCORRECT. Note 
        that sometimes, the original answer is not the best answer, and sometimes the chatbot's answer is not the 
        best answer. You are evaluating the chatbot's answer only. Example Format:\nQUESTION: question here\nCHATBOT 
        ANSWER: student's answer here\nORIGINAL ANSWER: original answer here\nGRADE: CORRECT or INCORRECT here\nPlease 
        remember to grade them based on being factually accurate. Begin!"""
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    return chat_prompt
