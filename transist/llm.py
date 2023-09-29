import json
import logging

from langchain.llms import OpenAI


log = logging.getLogger(__name__)


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
