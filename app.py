import json
import random

import langchain
from dotenv import load_dotenv
import gradio as gr
import logging
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate
)
import pydantic.v1.error_wrappers
from typing import Any, Dict, Tuple

from transist.llm import create_openai_llm, parse_json_maybe_invalid, ExtractionOutputParser
from transist.prompt import system_prompt, draft_question_prompt, extract_facts_prompt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

thinking = [
    "Give me a few seconds to understand what you told me.",
    "Let me take a moment to process the information you've shared",
    "Please allow me a short pause to fully comprehend the details you provided."
]
sufficient_facts_response = "Sufficient facts to generate section {section}"
move_to_next_section = "Let's proceed by moving on to the next section about {section}"


class CarbonAssistant(object):

    section_order = [
        (0, "info"),
        (2, "methodology"),
        (3, "quantification"),
        (4, "monitoring"),
        (5, "safeguards"),
        (1, "details"),
        (99, "closing")
    ]

    def __init__(self):
        self.state = "extract"
        self.sector = "afolu"
        self.extract_parser = ExtractionOutputParser()
        self.curr_section_index = 0
        self.curr_section_facts: Dict = self._load_section_facts(self.curr_section_index)
        self.completed_section: Dict[Tuple, Dict] = {}
        self.curr_questions = []

    def _load_section_facts(self, section_index):
        section_template = self._section_template(section_index)
        return json.loads(section_template)

    def _section_template(self, section_index):
        section_number, section = CarbonAssistant.section_order[section_index]
        section_dir = f"{section_number:02d}_{section}"
        section_file = f"{section_number:02d}_{self.sector}_{section}.json"
        filepath = f"data/templates/sector={self.sector}/{section_dir}/{section_file}"
        log.info("Getting template for %s from file: %s", section, filepath)
        return open(filepath, "r").read()

    @property
    def curr_section(self):
        return CarbonAssistant.section_order[self.curr_section_index][1]

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
            questions = self.draft_questions(llm, self.curr_section_facts)
            if self.sufficient_to_generate(questions):
                yield sufficient_facts_response % self.curr_section
                self.complete_section()
                if not self.next_section():
                    self.state = "generate"
                    yield "Generating document sections"
                else:
                    self.state = "draft"
                    yield move_to_next_section % self.curr_section
                    for out in self.design_with_llm(llm, message, history):
                        yield out
            else:
                self.curr_questions = questions
                self.state = "extract"
                yield "Let's continue gathering information about your carbon project"
                yield questions
        elif self.state == "extract":
            yield f"Thank you for providing information about your project. {random.choice(thinking)}"
            extracted = self.extract_facts(llm, message, history, self.curr_section_facts)
            if extracted.get("keys_updated", []):
                extracted_facts = extracted.get("extracted_project_facts", {})
                self.curr_section_facts.update(extracted_facts)
                log.info("Updated facts doc: %s", self.curr_section_facts)
                self.state = "draft"
            else:
                self.state = "explore"

            for out in self.design_with_llm(llm, message, history):
                yield out
        elif self.state == "explore":
            yield "I understand that you need some help in answering these questions."
            yield "Give me a moment to try and find some relevant information which can help."
            explore_results = self.explore(llm, message, history, self.curr_section_facts)
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

    def complete_section(self):
        self.curr_questions = []
        curr_section = CarbonAssistant.section_order[self.curr_section_index]
        if curr_section in self.completed_section:
            completed_facts = self.completed_section.get(curr_section)
            completed_facts.update(self.curr_section_facts)
        else:
            self.completed_section[curr_section] = self.curr_section_facts

    def next_section(self) -> bool:
        if self.curr_section_index + 1 >= len(CarbonAssistant.section_order):
            self.curr_section_facts = {}
            return False
        else:
            assert (0, "info") in self.complete_section(), \
                "Cannot move to next section without completing project info"
            self.curr_section_index += 1
            self.curr_section_facts = self._load_section_facts(self.curr_section_index)
            project_info_facts = self.completed_section[(0, "info")]
            self.curr_section_facts.update(project_info_facts)
            return True


def main():
    langchain.verbose = True
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
