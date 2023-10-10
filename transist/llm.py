import json
import logging
from typing import Any, Dict

from langchain.llms import OpenAI
from langchain.output_parsers.json import parse_and_check_json_markdown
from langchain.schema import BaseOutputParser, OutputParserException

log = logging.getLogger(__name__)


def create_openai_llm(openai_api_key=None):
    kwargs = dict(temperature=0, max_tokens=2400)
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


class ExtractionOutputParser(BaseOutputParser[Dict[str, str]]):
    """Parse for output of extraction chain."""

    def parse(self, text: str) -> Dict[str, Any]:
        try:
            expected_keys = ["extracted_project_facts", "keys_updated"]
            parsed = parse_and_check_json_markdown(text, expected_keys)
            if not isinstance(parsed["extracted_project_facts"], object):
                raise ValueError("Expected 'extracted_project_facts' to be an object")
            if not isinstance(parsed["keys_updated"], list):
                raise ValueError("Expected 'keys_updated' to be a list")

            return parsed
        except Exception as e:
            raise OutputParserException(
                f"Parsing text=\n{text}\n raise following error: \n{e}"
            ) from e