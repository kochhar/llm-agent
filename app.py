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

from transist.llm import create_openai_llm, parse_json_maybe_invalid, ExtractionOutputParser
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
        self.state = "extract"
        self.extract_parser = ExtractionOutputParser()
        self.waste_management_baseline_json = open("waste_management_baseline_template.json", "r").read()
        self.baseline_facts: Dict = json.loads(self.waste_management_baseline_json)
        self.sufficient_facts_response = "Sufficient facts to generate"
        self.current_questions = []

    def design(self, message, history, openai_api_key=None):
        try:
            llm = create_openai_llm(openai_api_key)
            for out in self.design_with_llm(llm, message, history):
                yield out
        except pydantic.v1.error_wrappers.ValidationError as e:
            if any(["OPENAI_API_KEY" in error['msg']for error in e.errors()]):
                raise gr.Error("An OpenAI API key needs to be provided in the Additional Inputs section below")
            else:
                raise gr.Error(pydantic.v1.error_wrappers.display_errors(e.errors()))

    def design_with_llm(self, llm, message, history):
        if self.state == "draft":
            questions = self.draft_questions(llm, self.baseline_facts)
            if self.sufficient_to_generate(questions):
                self.current_questions = []
                self.state = "generate"
                yield self.sufficient_facts_response
            else:
                self.current_questions = questions
                self.state = "extract"
                yield "Let's continue gathering information about your carbon project"
                yield questions
        elif self.state == "extract":
            yield f"Thank you for providing information about your project. {random.choice(thinking)}"

            extracted = self.extract_facts(llm, message, history, self.baseline_facts)
            if extracted.get("keys_updated", []):
                extracted_facts = extracted.get("extracted_project_facts", {})
                self.baseline_facts.update(extracted_facts)
                log.info("Updated facts doc: %s", self.baseline_facts)
                self.state = "draft"
            else:
                self.state = "explore"

            for out in self.design_with_llm(llm, message, history):
                yield out
        elif self.state == "explore":
            yield "I understand that you need some help in answering these questions."
            yield "Give me a moment to try and find some relevant information which can help."
            explore_results = self.explore(llm, message, history, self.baseline_facts)
            self.state = "extract"
            yield explore_results

    def draft_questions(self, llm, facts_document):
        questions_chain = LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_messages([system_prompt, draft_question_prompt]),
            output_key="questions",
            verbose=True)
        questions = questions_chain.predict(json_template=json.dumps(facts_document))
        return questions

    def extract_facts(self, llm, message, history, facts_document) -> Dict[Any, Any]:
        extract_chain = LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_messages([system_prompt, extract_facts_prompt]),
            output_parser=self.extract_parser,
            output_key="extracted",
            verbose=True)

        extracted: Dict[str, Any] = extract_chain.predict_and_parse(
            project_facts_document=json.dumps(facts_document),
            project_information=message)

        if not extracted:
            log.warning("Could not extracted using extract chain: '%s'", extracted)

        return extracted

    def explore(self, llm, message, history, facts_document):
        return f"""Some relevant search results to\n\nUser: {message}
        
        In context of \nhistory: 
        {history}"""

    @staticmethod
    def sufficient_to_generate(drafted_questions) -> bool:
        return drafted_questions.strip() == "GENERATE"


def main():
    assistant = CarbonAssistant()
    openai_api_key = gr.Textbox(placeholder="Please enter you OpenAI API key here",
                                label="Open AI API Key", render=False)
    demo = gr.ChatInterface(
        title="Verra Carbon Project Design Assistant",
        description="""
        I'm a virtual assistant who can help you in writing the baseline section for 
        your carbon project to be registered with the Verra registry. Please start by 
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
    load_dotenv()
    main()
