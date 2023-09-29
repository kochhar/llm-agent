from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

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
