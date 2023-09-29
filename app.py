import json
import random

from dotenv import load_dotenv
import gradio as gr
import logging
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate
)
import pydantic.v1.error_wrappers
from typing import Any, Dict

from transist.llm import create_llm, parse_json_maybe_invalid
from transist.prompt import system_prompt, draft_question_prompt, extract_facts_prompt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

thinking = [
    "Give me a few seconds to understand what you told me.",
    "Let me take a moment to process the information you've shared",
    "Please allow me a short pause to fully comprehend the details you provided."
]


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
    main()
