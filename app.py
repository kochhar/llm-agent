import json
import random

from dotenv import load_dotenv
import gradio as gr
import logging
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
import pydantic.v1.error_wrappers
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

system_prompt_template = """
You are a helpful and truthful carbon project design assistant who specialises in 
creating project design documents for carbon credit projects.

Your goal is to help project developers make a project design document for a carbon 
credit project to be registered with the CDM registry. 

You follow formatting instructions very carefully. When you are asked to create JSON 
output you ensure that the output is syntactically valid JSON. You also ensure it 
has no extra text before the start of the JSON instance and no extra text after the
end of the JSON instance.

"""
system_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)

section_prompt_template = """
I want to draft a section of the document which establishes and describes the 
baseline for the project. To write the section some essential factual details
about the project will need to be collected.

"""
section_prompt = HumanMessagePromptTemplate.from_template(section_prompt_template)

draft_question_prompt_template = """
The following JSON format is partially completed and lists the factual details 
essential for writing the project baseline section.


#### JSON FORMAT
{json_template}

----

Some elements of the JSON are incomplete and need to be filled in. Based on the 
incomplete elements of the JSON, I want you to ask me three of the most important
questions to collect missing information to be used for filling in incomplete
entries in the JSON. ONLY ask questions related to incomplete or missing 
information. DO NOT ask questions about elements which are complete.
"""
draft_question_prompt = HumanMessagePromptTemplate.from_template(section_prompt_template +
                                                                 draft_question_prompt_template)

extract_facts_prompt_template = """
I want you to extract factual details from information provided about a carbon 
project to update the JSON instance below. It is very import ONLY extract factual 
details from the project information.


#### JSON INSTANCE
{project_facts_document} 


#### Project Information
{project_information}

-----

Please extract factual details from the project information above and update the 
JSON Instance. It is very important to ONLY include factual information from the 
project information provided above.
  

The output MUST be formatted as a valid JSON instance. 
The output should ONLY contain the JSON instance with no other text before or after.
"""
extract_facts_prompt = HumanMessagePromptTemplate.from_template(section_prompt_template +
                                                                extract_facts_prompt_template)


thinking = [
    "Give me a few seconds to understand what you told me.",
    "Let me take a moment to process the information you've shared",
    "Please allow me a short pause to fully comprehend the details you provided."
]


def create_llm(openai_api_key=None):
    kwargs = dict(temperature=0, max_tokens=1536)
    if openai_api_key:
        kwargs['openai_api_key'] = openai_api_key
    return OpenAI(**kwargs)


def parse_json_maybe_invalid(json_as_str: str):
    try:
        return json.loads(json_as_str)
    except json.JSONDecodeError:
        first_brace, last_brace = json_as_str.find('{'), json_as_str.rfind('}')
        last_brace = last_brace + 1 if not last_brace == -1 else len(json_as_str)
        stripped_string = json_as_str[first_brace:last_brace]
        try:
            return json.loads(stripped_string)
        except json.JSONDecodeError as e:
            log.error("Could not parse extracted facts string '%s' as JSON. Error: %s", json_as_str, e)
            return {}


class CarbonAssistant(object):
    def __init__(self):
        self.llm = None
        self.state = "extract"
        self.waste_management_baseline_json = open("waste_management_baseline_template.json", "r").read()
        self.waste_management_baseline_facts: Dict = json.loads(self.waste_management_baseline_json)
        self.sufficient_facts_response = "Sufficient facts to generate"

    def design(self, message, history, openai_api_key=None):
        try:
            self.llm = create_llm(openai_api_key)
        except pydantic.v1.error_wrappers.ValidationError as e:
            if any(["OPENAI_API_KEY" in error['msg']for error in e.errors()]):
                raise gr.Error("An OpenAI API key needs to be provided in the Additional Inputs section below")
            else:
                raise gr.Error(pydantic.v1.error_wrappers.display_errors(e.errors()))

        if self.state == "draft":
            questions = self.draft_questions(self.waste_management_baseline_facts)
            if self.sufficient_to_generate(questions):
                self.state = "generate"
                yield self.sufficient_facts_response
            else:
                self.state = "extract"
                yield "Let's continue gathering information about your carbon project"
                yield questions
        elif self.state == "extract":
            yield f"Thank you for providing information about your project. {random.choice(thinking)}"

            extracted = self.extract_facts(message, history, self.waste_management_baseline_facts)
            self.waste_management_baseline_facts.update(extracted)
            log.info("Updated facts doc: %s", self.waste_management_baseline_facts)
            self.state = "draft"
            for out in self.design(message, history, openai_api_key):
                yield out

    def draft_questions(self, facts_document):
        questions_chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([system_prompt, draft_question_prompt]),
            output_key="questions",
            verbose=True)
        questions = questions_chain.predict(json_template=json.dumps(facts_document))
        return questions

    def extract_facts(self, message, history, facts_document) -> Dict[Any, Any]:
        extract_chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([system_prompt, extract_facts_prompt]),
            output_key="extracted_facts",
            verbose=True)

        extracted_facts_str = extract_chain.predict(project_facts_document=json.dumps(facts_document),
                                                    project_information=message)
        if not extracted_facts_str:
            log.warning("Could not get extracted facts from extract chain: '%s'", extracted_facts_str)
            extracted_facts_str = "{}"

        return parse_json_maybe_invalid(extracted_facts_str)

    @staticmethod
    def sufficient_to_generate(drafted_questions) -> bool:
        return drafted_questions.strip() == "GENERATE"


def main():
    assistant = CarbonAssistant()
    openai_api_key = gr.Textbox(placeholder="Please enter you OpenAI API key here",
                                label="Open AI API Key", render=False)
    demo = gr.ChatInterface(
        title="CDM PDD Baseline Assistant",
        description="""
        I'm a virtual assistant who can help you in writing the baseline section for 
        your carbon project to be registered with the CDM registry. Please start by 
        telling me something about your project.
        """,
        textbox=gr.Textbox(placeholder="Start by telling me about your project",
                           scale=7),
        fn=assistant.design,
        additional_inputs=[openai_api_key],
        examples=[["The name of my project is BrewHat Bunguluru Waste Management", None],
                  ["My project falls under the Waste Management sectoral scope", None],
                  ["My project is about reducing GHG emission from biomass waste", None]]
    )
    demo.queue().launch()


if __name__ == "__main__":
    # Take environment variables from .env file
    print(load_dotenv())
    main()
