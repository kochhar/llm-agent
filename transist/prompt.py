from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate


system_prompt_template = """\
You are a helpful and truthful carbon project design assistant specialising in \
creating project design documents for carbon credit projects.

Your goal is to help project developers make a project design document for a \
carbon credit project to be registered with the Verra registry.

You follow formatting instructions very carefully.
"""
system_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)


section_prompt_template = """
I want to draft a section of the document which establishes and describes the \
baseline for the project. To write the section some essential factual details \
about the project will need to be collected.

"""
section_prompt = HumanMessagePromptTemplate.from_template(section_prompt_template)


draft_question_prompt_template = """\
The following project facts document lists the factual details essential for \
writing the project baseline section. The document is partially completed and \
and some elements need to be filled in.

Based on the incomplete elements of the project facts document, I want you to \
ask three most important questions to collect missing information. The answers \
will be used for filling in incomplete entries in the project facts document.

REMEMBER: Only ask questions related to incomplete or missing information. DO \
NOT ask questions about elements which are already complete.

<< PROJECT FACTS DOCUMENT >>
{json_template}
"""
draft_question_prompt = HumanMessagePromptTemplate.from_template(section_prompt_template +
                                                                 draft_question_prompt_template)


extract_facts_prompt_template = """\
Given raw text input containing information about a carbon project, extract \
the factual details about the project and update the project facts document \
provided as input. You will be given the project developer's input and a \
partially filled project facts document. 

<< FORMATTING >>
Return a markdown code snippet containing a JSON object
```json
{{{{
    "updated_project_facts": object \\ the updated project facts,
    "keys_updated": array \\ a list of updated top-level keys in updated_project_facts
}}}}
```

REMEMBER: It is very import to ONLY extract factual details from the project \
information. If no factual details can be extracted "updated_project_facts" \
should be a copy of the project facts.
REMEMBER: "keys_updated" must contain all the top-level keys which are \ 
modified. If no factual details can be extracted "keys_updated" should be empty.

<< PROJECT FACTS DOCUMENT >>
{project_facts_document} 

<< INPUT >>
{project_information}

Now please extract factual details from the input above and update the project \
facts document. It is ESSENTIAL to ONLY include factual information from the \
input provided.

<< OUTPUT (must include ```json at the start of the response) >>
<< OUTPUT (must end with ```) >>
"""
extract_facts_prompt = HumanMessagePromptTemplate.from_template(section_prompt_template +
                                                                extract_facts_prompt_template)
