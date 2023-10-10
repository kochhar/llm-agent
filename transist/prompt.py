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
The following project facts document lists the details essential for writing \
the project baseline section. The document is partially completed and and \
some elements need to be filled in.

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
Given raw input containing information about a carbon project, extract factual \
details about the project and update the project facts document provided. You \
will be given the project developer's input and a partially filled project \
facts document. 

<< FORMATTING >>
Return a markdown snippet containing the extract project facts and the keys \
which were updated.
```json
{{{{
    "keys_updated": array \\ a list of updated top-level keys in updated_project_facts
    "extracted_project_facts": object \\ the project facts document with updated \
    information
}}}}
```

REMEMBER: It is extremely important to ONLY extract factual details from the \
input provided. If no factual details can be extracted "extracted_project_facts" \
should be empty.
REMEMBER: It is important for the "extracted_project_facts" output to include \
ONLY the updated top-level keys and nested sub-keys. Keys and values of of \
the project which are unchanged can be omitted from the output. 
REMEMBER: "keys_updated" must contain all the top-level keys which are \
modified. If no factual details can be extracted "keys_updated" should be empty

<< PROJECT FACTS DOCUMENT >>
{project_facts_document} 

<< INPUT >>
{project_information}

Now please extract factual details from the input. It is ESSENTIAL to ONLY \
include factual information from the input provided.

<< OUTPUT (must include ```json at the start of the response) >>
<< OUTPUT (must end with ```) >>
"""
extract_facts_prompt = HumanMessagePromptTemplate.from_template(section_prompt_template +
                                                                extract_facts_prompt_template)
