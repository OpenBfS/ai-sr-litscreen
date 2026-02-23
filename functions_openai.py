from pydantic import BaseModel, Field, field_validator
from typing import Literal
import json
from langchain.schema import AIMessage
import re
import pandas as pd
import pickle
from openai import OpenAI
import time

def clean_json_string(json_string):
    ''' We will need this function to clean up erratic LLM responses '''
    # json_string = json_string.replace('\\n', ' ')

    clean_string = []
    in_string = False
    prev_char = ''
    prev_openquote = ''

    i = 0
    while i < len(json_string):
        char = json_string[i]

        # String manipulation
        if (char == "'" or char == '"') and not in_string:
            in_string = True
            clean_string.append('"')
            prev_openquote = char
        
        elif char == prev_openquote and in_string:
            in_string = False
            clean_string.append('"')

        elif in_string:
            clean_string.append(char)

        elif char == '/' and i + 1 < len(json_string) and json_string[i + 1] == '/' and not in_string:
            while i < len(json_string) and json_string[i] != '\n':
                i += 1

        elif char == '#' and not in_string:
            while i < len(json_string) and json_string[i] != '\n':
                i += 1

        
        elif char in ('{', '}', '[', ']', ':', ','):
            # Outside string, only keep structural characters
            clean_string.append(char)

        prev_char = char
        i += 1
    
    # Join the cleaned parts into a new JSON string
    clean_string = ''.join(clean_string)
    # Remove any trailing commas before closing brackets
    # clean_string = re.sub(r',\s*([\]}])', r'\1', clean_string)
    return clean_string

def extract_and_parse_json(input_content):
    ''' we will need this to extract json content from a malformatted LLM response'''
    # Find the first occurrence of '{'
    first_brace_index = input_content.find('{')
    last_brace_index = input_content.rfind('}')
    # Check if '{' was found; if not, we cannot have valid JSON#
    if first_brace_index == -1:
        print("No JSON object detected in the input content.")
        return {input_content}
    # Check if the closing brace was found
    if last_brace_index == -1:
        print("Incomplete JSON object detected in the input content. Adding closing '}'.")
        input_content += '}'
    # Prune everything before the first '{'
    json_string = input_content[first_brace_index:last_brace_index + 1]
    # return json_string # for debugging purposes
    # Clean the JSON string (e.g., to remove comments)
    try:
        cleaned_json_string = clean_json_string(json_string)

        return cleaned_json_string
    except Exception as e:
        print("Manual closing of json string failed due to :", e)
        return {input_content}


def run_batch(method, savedir_batch, abstracts_dict, system_prompt, tools, model="gpt-4.1-mini-2025-04-14", job_description="Batch job for processing abstracts with OpenAI API", index=None, placeholder=None, monitor=False, recheck_time=120,temperature=0.0, max_tokens=512, reasoning_model = False, reasoning_effort = None, examples=None, previous_results=None, previous_extract=False, extract = False, omit_abstract = False):
    """
    This function runs a batch job using the OpenAI API.
    It retrieves the results from the API and processes them.
    :param method: The method to be used for the batch job.
    :param abstracts_dict: A list of dictionaries containing the abstracts to be processed.
    :param system_prompt: The system prompt to be used for the batch job.
    :param tools: A list of tools to be used in the batch job.
    :param model: The model to be used for the batch job. Default is "gpt-4.1-nano".
    :param job_description: A description of the batch job.
    :param index: An optional index or list of indices to specify which abstracts to process.
    :param placeholder: An optional placeholder for search results.
    :param monitor: A boolean indicating whether to monitor the batch job status.
    :param recheck_time: Time in second after which the status of the batch job is retrieved again. Only relevant when monitor=True.
    :param temperature: The temperature to be used for the batch job. Default is 0.0.
    :param max_tokens: The maximum number of tokens to be used for the batch job. Default is 512.
    :param reasoning_model: A boolean indicating whether to use a reasoning model.
    :param reasoning_effort: The reasoning effort to be used for the batch job. Default is None.
    :param examples: Optional examples to be used in the batch job.
    :param previous_results: Optional previous results to be used in the batch job.
    :param previous_extract: A boolean indicating whether features were extracted in a previous run and to be provided here.
    :param extract: A boolean indicating whether to extract features from the abstracts. Only relevant if examples are provided.
    :param omit_abstract: A boolean indicating whether to omit the abstract text in the prompt.
    :return: The name of the file to save the batch job results to.
    """


    # Creating an array of json tasks
    method=method
    tasks = []
    
    if index is None:
        index = range(len(abstracts_dict))
    elif isinstance(index, list):
        index = index

    elif isinstance(index, int):
        index = [index]
    else:
        raise ValueError("Index must be an integer, a list of integers, or None.")
    
    for i in index:

        abs = abstracts_dict[i]
        # If the abstract is a string, convert it to a dictionary
        if isinstance(abs, str):
            abs = {"input": abs}
        elif not isinstance(abs, dict):
            raise ValueError("Each abstract must be a string or a dictionary.")
        
        description = f"Study text: {abs['input']}"
        if placeholder is not None:
            description = f"{description} \n\n {placeholder[i]}"
        elif examples is not None:
            if extract is False:
                description = f"Examples: {examples} \n\n Now, it is your turn to evaluate the following study text: {abs['input']} \n You MUST call and use the ValidateForInclusion tool to format your response."
            elif extract is True:
                description = f"Examples: {examples} \n\n Now it is you turn to extract the five study features from the following study text: {abs['input']} \n You must provide a structured response output as in the examples."
        elif previous_results is not None and previous_extract is False:
            description = f"Study text: {abs['input']} \n\nPrevious Reviewer's Decisions: {previous_results[i]} . "
        elif previous_extract is True and previous_results is not None and omit_abstract is False:
            description = f"""Study text: {abs['input']} \n
                            Description of population(s): {previous_results[i]['population']}
                            Description of exposure(s): {previous_results[i]['exposure']}
                            Description of comparison(s): {previous_results[i]['comparison']}
                            Description of outcome(s): {previous_results[i]['outcome']}
                            Description of study type(s): {previous_results[i]['study_type']}"""
        elif previous_extract is True and previous_results is not None and omit_abstract is True:
            description = f"""Description of population(s): {previous_results[i]['population']}
                            Description of exposure(s): {previous_results[i]['exposure']}
                            Description of comparison(s): {previous_results[i]['comparison']}
                            Description of outcome(s): {previous_results[i]['outcome']}
                            Description of study type(s): {previous_results[i]['study_type']}"""

        task = {
            "custom_id": f"{method}-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                    # This is what you would have in your Chat Completions API call
                    "model": model,  # gpt-4.1-nano-2025-04-14
                    # "temperature": temperature,  # commented out since we need to define in dependece on reasoning_model request
                    # "max_tokens": max_tokens,   # for v1/responses "max_output_tokens"
                    # "reasoning_effort": reasoning_effort, # commented out since we need to define in dependece on reasoning_model request
                    # "response_format": { "type": "json_schema", "json_schema": json_schema }, # unneccessary inflation of input tokens, tool definition is sufficient # for v1/responses "text_format" allegeder
                    "messages": [   # for v1/responses "input"
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": description
                        }
                    ],
                    "tool_choice" : "required",
                    "tools" : tools
                }
        }

        if reasoning_model:
            task["body"]["max_completion_tokens"] = max_tokens
            task["body"]["reasoning_effort"] = reasoning_effort
        else:
            task["body"]["max_tokens"] = max_tokens
            task["body"]["temperature"] = temperature
            # task["body"]["parallel_tool_calls "] = False    # uncomment this line to prevent parallel calls in the output by non-reasoning models.
        tasks.append(task)

    # saving tasks to file
    file_name = f"{savedir_batch}/tasks_{method}.jsonl"

    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')

    client = OpenAI()

    batch_file = client.files.create(
      file=open(file_name, "rb"),
      purpose="batch"
    )
    batch_job = client.batches.create(
      input_file_id=batch_file.id,
      endpoint="/v1/chat/completions",
      completion_window="24h",
      metadata={"description": job_description},
    )

    if monitor:
        print(f"Batch job {batch_job.id} created. Monitoring status...")
        # monitor completion status
        while True:
            batch_job = client.batches.retrieve(batch_job.id)
            if batch_job.status == "completed":
                print(f"job {batch_job.id} is done")
                break
            elif batch_job.status == "failed":
                print(f"job {batch_job.id} failed")
                print(batch_job.errors)
                break
            else:
                time.sleep(recheck_time)
                print(f"job {batch_job.id} is still running")
        
        # examine success of batch, then save and load the results into lists

        if batch_job.error_file_id:
            print("Batch job failed.")
            file_response = client.files.content(batch_job.error_file_id)
            print(file_response.text)

        elif batch_job.output_file_id:
            print("Batch job completed successfully.")
            print(f"Output file ID: {batch_job.output_file_id}")
            result_file_id = batch_job.output_file_id
            result = client.files.content(result_file_id).content

            result_file_name = f"{savedir_batch}/results/batch_{method}.jsonl"

            with open(result_file_name, 'wb') as file:
                print("Saving results to: ", result_file_name)
                file.write(result)
            return result_file_name, batch_job.id
            
    else:
        print(f"Batch job {batch_job.id} created. No monitoring. You can check the status later using the OpenAI API. Use the file ID {batch_job.id} to retrieve the results and save them to a file name as returned by this function.")
        result_file_name = f"{savedir_batch}/results/batch_{method}.jsonl"
    # Wait for the batch job to complete and save the results
    return result_file_name, batch_job.id

def fetch_batch_results(batch_job_id: str, result_file_name: str):
    """
    Fetches the results of a batch job id from openAI.
    :param batch_job_id: The name of the job id containing the batch job results.
    :param result_file_name: The name of the file to save the batch job results to.
    :raises ValueError: If the batch job is not completed or has no output file.
    :raises ValueError: If the batch job has no output file.
    :return: A list of dictionaries containing the batch job results.
    """
    client = OpenAI()
    batch_job = client.batches.retrieve(batch_job_id)
    if batch_job.status not in ["completed", "expired"]:
        raise ValueError(f"Batch job {batch_job_id} is not completed. Current status: {batch_job.status}")
    if not batch_job.output_file_id:
        raise ValueError(f"Batch job {batch_job_id} has no output file. Current status: {batch_job.status}")
    else:
        if batch_job.status == "expired":
            print("Batch job expired, but could be downloaded. Be careful and check if results are complete.")
        else:
            print("Batch job completed successfully.")
            print(f"Output file ID: {batch_job.output_file_id}")
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content
    # Save the results to a file

    with open(result_file_name, 'wb') as file:
        print("Saving results to: ", result_file_name)
        file.write(result)


class BatchResults:
    # define a class to hold the results of the batch job compatible with the individually engineered parsing function (see def get_message_zeroshot)
    """A class to hold the results of a batch job."""
    def __init__(self, tool_call, tool_name, content, custom_id):
        self.tool_call = tool_call
        self.tool_name = tool_name
        self.content = content
        self.custom_id = custom_id

def extract_batchAPI(result_file_name):
    """
    This function extracts the results from a batch API call saved in a file.
    It reads the file line by line, parses each line as a JSON object, and extracts relevant information.
    It returns lists of results, including the full JSON objects, extracted tool calls, content, and custom IDs.
    :param result_file_name: The name of the file containing the batch API results.
    :return: A tuple containing lists of results_index, results_full, results_extracted, results_content, and results.
    """
    # Loading data from saved file
    results_full = []
    results_extracted = []
    results_index = []
    results_content = []
    results = []

    with open(result_file_name, 'r') as file:
        for line in file:
            # make sure that all objects are reset with each loop iteration
            tool_call = {}
            tool_calls = None
            tool_name = set()
            json_object = None
            arguments_dict = None
            custom_id = ''
            content = ''
            
            # Parsing the JSON string into a dict and appending to the list of results
            json_object = json.loads(line.strip())
            try:
                tool_calls = json_object["response"]["body"]["choices"][0]["message"]['tool_calls']
            except Exception as e:
                tool_calls = None
            # Process each tool call if they exist
            
            if tool_calls:
                # This loop is designed in a way that the last tool_call will be used for results. This is important because sometimes models make parallel tool calls with the same tool, resulting in redundant results. The parameter parallel_tool_calls can be set to False (Default is True) in the batch job function, but reportedly not all models accept this parameter.
                for call in tool_calls:
                    # Parse the arguments string into a dictionary
                    try:
                        arguments_dict = json.loads(call["function"]["arguments"])
                    except Exception as e:
                        print(e, "\n", call["function"]["arguments"])
                    # Update the tool_dict with the key-value pairs from arguments
                    if arguments_dict:
                        tool_call.update(arguments_dict)

                    tool_name.add(call["function"]["name"])

            if len(tool_name) == 1:
                tool_name = list(tool_name)[0]
            else:
                tool_name = list(tool_name)

            custom_id = json_object["custom_id"]
            try:
                content = json_object["response"]["body"]["choices"][0]["message"]["content"]
                if not content:
                    content = ''
            except Exception as e:
                content = ''

            results.append(BatchResults(tool_call, tool_name, content, custom_id))

            results_full.append(json_object)

            results_extracted.append(tool_call)

            results_content.append(json_object["response"]["body"]["choices"][0]["message"]["content"])

            results_index.append(json_object["custom_id"])

    # Make sure that the order of the results is correct and aligns with the dataset.
    # Create a DataFrame, sort by custom_id, and transfer back to lists.
    df = pd.DataFrame({
        'custom_ids': results_index,
        'results_full': results_full,
        'results_extracted': results_extracted,
        'results_content': results_content,
        'class_results': results
    })

    
    # Define a key function that extracts the last numeric sequence from the string 
    def sort_by_trailing_number(text):
        match = re.search(r'(\d+)$', text)
        return int(match.group(1)) if match else float('inf')  # or 0 if you prefer

    # Check if the order is already correct
    order_is_correct = results_index == sorted(df['custom_ids'], key=sort_by_trailing_number)

    if not order_is_correct:
        print("Order was not maintained.")
        # Sort the DataFrame based on the trailing numbers
        df = df.sort_values(by='custom_ids', key=lambda col: col.map(sort_by_trailing_number))
        order_is_correct = results_index == sorted(df['custom_ids'], key=sort_by_trailing_number)
        if order_is_correct:
            print("Order was fixed.")

    # Extract the sorted lists back
    results_index = df['custom_ids'].tolist()
    results_full = df['results_full'].tolist()
    results_extracted = df['results_extracted'].tolist()
    results_content = df['results_content'].tolist()
    results = df['class_results'].tolist()
    
    return results_index, results_full, results_extracted, results_content, results

class ValidateForInclusion(BaseModel):
    population_reason: str = Field(
        description="Short explanation of the reasoning for the evaluation result of study population (max 30 words)"
    )
    population_decision: Literal['True', 'False', 'NotEvaluable'] = Field(
        description="Evaluation result of the study population: True (include), False (exclude), or NotEvaluable"
    )


    exposure_reason: str = Field(
        description="Short explanation of the reasoning for the evaluation result of exposure (max 30 words)"
    )
    exposure_decision: Literal['True', 'False', 'NotEvaluable'] = Field(
        description="Evaluation result of the exposure: True (include), False (exclude), or NotEvaluable"
    )


    comparison_reason: str = Field(
        description="Short explanation of the reasoning for the evaluation result of comparison (max 30 words)"
    )
    comparison_decision: Literal['True', 'False', 'NotEvaluable'] = Field(
        description="Evaluation result of the comparison: True (include), False (exclude), or NotEvaluable"
    )


    outcome_reason: str = Field(
        description="Short explanation of the reasoning for the evaluation result of outcome (max 30 words)"
    )
    outcome_decision: Literal['True', 'False', 'NotEvaluable'] = Field(
        description="Evaluation result of the outcome: True (include), False (exclude), or NotEvaluable"
    )


    study_type_reason: str = Field(
        description="Short explanation of the reasoning for the evaluation result of study type (max 30 words)"
    )
    study_type_decision: Literal['True', 'False', 'NotEvaluable'] = Field(
        description="Evaluation result of the study type: True (include), False (exclude), or NotEvaluable"
    )


    @field_validator('population_reason', 'exposure_reason', 'comparison_reason', 'outcome_reason', 'study_type_reason')
    def validate_reason_length(cls, v):
        words = v.split()
        if len(words) > 30:
            raise ValueError('Reason must not exceed 30 words')
        return v


def get_message(input):
    """
    Need a parse wrapper for the PECOS validation LLM output.
    :param input: LLM response message
    :return: AIMessage with properly formatted function_call arguments
    """
    blank_msg = AIMessage(content=input.content, additional_kwargs={'function_call' : {'name': 
                                                                'ValidateForInclusion', 'arguments' : 
                                                                json.dumps(
                                                                    {'population_reason': '', 'population_decision': 'NotEvaluable',
                                                                    'exposure_reason': '', 'exposure_decision': 'NotEvaluable',
                                                                    'comparison_reason': '', 'comparison_decision': 'NotEvaluable',
                                                                    'outcome_reason': '', 'outcome_decision': 'NotEvaluable',
                                                                    'study_type_reason': '', 'study_type_decision': 'NotEvaluable'}
                                                                    )
                                                                }
                                                                }
                                                                )
    try:
        msg = AIMessage(content=input.content, additional_kwargs={'function_call' : {'name': 
                                                                        input.tool_name, 'arguments' : 
                                                                        json.dumps(input.tool_call)
                                                                        }
                                                                        }
                                                                        )
        arguments = json.loads(msg.additional_kwargs['function_call']['arguments'])
        for key in ['population_reason', 'population_decision', 'exposure_reason', 'exposure_decision', 'comparison_reason', 'comparison_decision', 'outcome_reason', 'outcome_decision', 'study_type_reason', 'study_type_decision']:
            if key not in arguments:
                print("Model failed to provide PECOS element in output: ", key)
                print(arguments)
                if "ion" in key[-5:]:
                    arguments[key] = 'NotEvaluable'
                    print("Inserted NE value for: ", key)
                else:
                    arguments[key] = ''
                    print("Inserted empty value for: ", key)
                msg.additional_kwargs['function_call']['arguments'] = json.dumps(arguments)                
        return msg
    except Exception as e:    # sometimes the model responds with 'I cannot help you' and fails to provide a proper reponse format, we need to work around it
        print("Encountered error: ", e, "Model refused to answer in appropriate format:", input.content)
        try:
            # Attempt to parse the JSON content
            torescue = extract_and_parse_json(input.content)
            rescue_format = json.loads(torescue)
            for key in ['parameters', 'arguments', 'args','response']:
            # Attempt to parse the JSON content
                if key in rescue_format.keys():
                    rescue_arguments = rescue_format[key]
                    print(rescue_arguments)
                    arguments = {'population_reason': '', 'population_decision': 'NotEvaluable',
                                                                    'exposure_reason': '', 'exposure_decision': 'NotEvaluable',
                                                                    'comparison_reason': '', 'comparison_decision': 'NotEvaluable',
                                                                    'outcome_reason': '', 'outcome_decision': 'NotEvaluable',
                                                                    'study_type_reason': '', 'study_type_decision': 'NotEvaluable'}
                    # check if the model provided all elements already:
                    for key in ['population_reason', 'population_decision', 'exposure_reason', 'exposure_decision', 'comparison_reason', 'comparison_decision', 'outcome_reason', 'outcome_decision', 'study_type_reason', 'study_type_decision']:
                        if key in rescue_arguments.keys():
                            arguments[key] = rescue_arguments[key] # update arguments
                    msg = AIMessage(content=input.content, additional_kwargs={'function_call' : {'name':
                                                                                        'ValidateForInclusion', 'arguments' : json.dumps(arguments)
                                                                                        }
                                                                                        }
                                                                                        )
                    return msg
            print(rescue_format)
            arguments = {'population_reason': '', 'population_decision': 'NotEvaluable', 'exposure_reason': '', 'exposure_decision': 'NotEvaluable', 'comparison_reason': '', 'comparison_decision': 'NotEvaluable','outcome_reason': '', 'outcome_decision': 'NotEvaluable','study_type_reason': '', 'study_type_decision': 'NotEvaluable'}
            # check if the model provided all elements already:
            for key in ['population_reason', 'population_decision', 'exposure_reason', 'exposure_decision', 'comparison_reason', 'comparison_decision', 'outcome_reason', 'outcome_decision', 'study_type_reason', 'study_type_decision']:
                if key in rescue_format.keys():
                    arguments[key] = rescue_format[key] # update arguments
            msg = AIMessage(content=input.content, additional_kwargs={'function_call' : {'name':
                                                                                'ValidateForInclusion', 'arguments' : json.dumps(arguments)
                                                                                }
                                                                                }
                                                                                )
            return msg
        except json.JSONDecodeError as e:
            # Log the error for debugging
            print("Model did not provide a properly formatted response. Likely JSON Decode Error:", e)
            
            # Attempt minor corrections
            if isinstance(input.content, str) and len(input.content) >= 2:
                if input.content[-2:] != '}}':
                    input.content += '}'
            
                try:
                    torescue = extract_and_parse_json(input.content)
                    rescue_format = json.loads(torescue)
                    for key in ['parameters', 'arguments', 'args', 'response']:
                        # Attempt to parse the JSON content
                        if key in rescue_format.keys():
                            rescue_arguments = rescue_format[key]
                            print(rescue_arguments)
                            arguments = {'population_reason': '', 'population_decision': 'NotEvaluable',
                                                                            'exposure_reason': '', 'exposure_decision': 'NotEvaluable',
                                                                            'comparison_reason': '', 'comparison_decision': 'NotEvaluable',
                                                                            'outcome_reason': '', 'outcome_decision': 'NotEvaluable',
                                                                            'study_type_reason': '', 'study_type_decision': 'NotEvaluable'}
                            # check if the model provided all elements already:
                            for key in ['population_reason', 'population_decision', 'exposure_reason', 'exposure_decision', 'comparison_reason', 'comparison_decision', 'outcome_reason', 'outcome_decision', 'study_type_reason', 'study_type_decision']:
                                if key in rescue_arguments.keys():
                                    arguments[key] = rescue_arguments[key] # update arguments
                            msg = AIMessage(content=input.content, additional_kwargs={'function_call' : {'name':
                                                                                                'ValidateForInclusion', 'arguments' : json.dumps(arguments)
                                                                                                }
                                                                                                }
                                                                                                )
                            print("Returning message as:", msg)     
                            return msg
                    return blank_msg   
                except Exception as e:
                    # print("Still unable to decode JSON:", e)
                    print("Model did not provide the correct format in the content window. Equating to null response. ", e)
                    return blank_msg    
            else:
                return blank_msg
        except Exception as e:
            print("Model did not provide the correct format in the content window. Equating to null response. ", e)
            return blank_msg
    
def get_message_langchain(input):
    blank_msg = AIMessage(content=input.content, additional_kwargs={'function_call' : {'name': 
                                                                'ValidateForInclusion', 'arguments' : 
                                                                json.dumps(
                                                                    {'population_reason': '', 'population_decision': 'NotEvaluable',
                                                                    'exposure_reason': '', 'exposure_decision': 'NotEvaluable',
                                                                    'comparison_reason': '', 'comparison_decision': 'NotEvaluable',
                                                                    'outcome_reason': '', 'outcome_decision': 'NotEvaluable',
                                                                    'study_type_reason': '', 'study_type_decision': 'NotEvaluable'}
                                                                    )
                                                                }
                                                                }
                                                                )
    try:
        msg = AIMessage(content=input.content, additional_kwargs={'function_call' : {'name': 
                                                                        input.tool_calls[0]['name'], 'arguments' : 
                                                                        json.dumps(input.tool_calls[0]['args'])
                                                                        }
                                                                        }
                                                                        )
        arguments = json.loads(msg.additional_kwargs['function_call']['arguments'])
        for key in ['population_reason', 'population_decision', 'exposure_reason', 'exposure_decision', 'comparison_reason', 'comparison_decision', 'outcome_reason', 'outcome_decision', 'study_type_reason', 'study_type_decision']:
            if key not in arguments:
                print("Model failed to provide PECOS element in output: ", key)
                print(arguments)
                if "ion" in key[-5:]:
                    arguments[key] = 'NotEvaluable'
                    print("Inserted NE value for: ", key)
                else:
                    arguments[key] = ''
                    print("Inserted empty value for: ", key)
                msg.additional_kwargs['function_call']['arguments'] = json.dumps(arguments)                
        return msg
    except Exception as e:    # sometimes the model responds with 'I cannot help you' and fails to provide a proper reponse format, we need to work around it
        print("Encountered error: ", e, "Model refused to answer in appropriate format:", input.content)
        try:
            # Attempt to parse the JSON content
            torescue = extract_and_parse_json(input.content)
            rescue_format = json.loads(torescue)
            for key in ['parameters', 'arguments', 'args','response']:
            # Attempt to parse the JSON content
                if key in rescue_format.keys():
                    rescue_arguments = rescue_format[key]
                    print(rescue_arguments)
                    arguments = {'population_reason': '', 'population_decision': 'NotEvaluable',
                                                                    'exposure_reason': '', 'exposure_decision': 'NotEvaluable',
                                                                    'comparison_reason': '', 'comparison_decision': 'NotEvaluable',
                                                                    'outcome_reason': '', 'outcome_decision': 'NotEvaluable',
                                                                    'study_type_reason': '', 'study_type_decision': 'NotEvaluable'}
                    # check if the model provided all elements already:
                    for key in ['population_reason', 'population_decision', 'exposure_reason', 'exposure_decision', 'comparison_reason', 'comparison_decision', 'outcome_reason', 'outcome_decision', 'study_type_reason', 'study_type_decision']:
                        if key in rescue_arguments.keys():
                            arguments[key] = rescue_arguments[key] # update arguments
                    msg = AIMessage(content=input.content, additional_kwargs={'function_call' : {'name':
                                                                                        'ValidateForInclusion', 'arguments' : json.dumps(arguments)
                                                                                        }
                                                                                        }
                                                                                        )
                    return msg
            print(rescue_format)
            arguments = {'population_reason': '', 'population_decision': 'NotEvaluable', 'exposure_reason': '', 'exposure_decision': 'NotEvaluable', 'comparison_reason': '', 'comparison_decision': 'NotEvaluable','outcome_reason': '', 'outcome_decision': 'NotEvaluable','study_type_reason': '', 'study_type_decision': 'NotEvaluable'}
            # check if the model provided all elements already:
            for key in ['population_reason', 'population_decision', 'exposure_reason', 'exposure_decision', 'comparison_reason', 'comparison_decision', 'outcome_reason', 'outcome_decision', 'study_type_reason', 'study_type_decision']:
                if key in rescue_format.keys():
                    arguments[key] = rescue_format[key] # update arguments
            msg = AIMessage(content=input.content, additional_kwargs={'function_call' : {'name':
                                                                                'ValidateForInclusion', 'arguments' : json.dumps(arguments)
                                                                                }
                                                                                }
                                                                                )
            return msg
        except json.JSONDecodeError as e:
            # Log the error for debugging
            print("Model did not provide a properly formatted response. Likely JSON Decode Error:", e)
            
            # Attempt minor corrections
            if isinstance(input.content, str) and len(input.content) >= 2:
                if input.content[-2:] != '}}':
                    input.content += '}'
            
                try:
                    torescue = extract_and_parse_json(input.content)
                    rescue_format = json.loads(torescue)
                    for key in ['parameters', 'arguments', 'args', 'response']:
                        # Attempt to parse the JSON content
                        if key in rescue_format.keys():
                            rescue_arguments = rescue_format[key]
                            print(rescue_arguments)
                            arguments = {'population_reason': '', 'population_decision': 'NotEvaluable',
                                                                            'exposure_reason': '', 'exposure_decision': 'NotEvaluable',
                                                                            'comparison_reason': '', 'comparison_decision': 'NotEvaluable',
                                                                            'outcome_reason': '', 'outcome_decision': 'NotEvaluable',
                                                                            'study_type_reason': '', 'study_type_decision': 'NotEvaluable'}
                            # check if the model provided all elements already:
                            for key in ['population_reason', 'population_decision', 'exposure_reason', 'exposure_decision', 'comparison_reason', 'comparison_decision', 'outcome_reason', 'outcome_decision', 'study_type_reason', 'study_type_decision']:
                                if key in rescue_arguments.keys():
                                    arguments[key] = rescue_arguments[key] # update arguments
                            msg = AIMessage(content=input.content, additional_kwargs={'function_call' : {'name':
                                                                                                'ValidateForInclusion', 'arguments' : json.dumps(arguments)
                                                                                                }
                                                                                                }
                                                                                                )
                            print("Returning message as:", msg)     
                            return msg
                    return blank_msg   
                except Exception as e:
                    # print("Still unable to decode JSON:", e)
                    print("Model did not provide the correct format in the content window. Equating to null response. ", e)
                    return blank_msg    
            else:
                return blank_msg
        except Exception as e:
            print("Model did not provide the correct format in the content window. Equating to null response. ", e)
            return blank_msg   


def format_zeroshot_results(zero_shot_result):
    # Combine all evaluation results into a single dictionary
    evaluation_dict = {}

    for key in ['population', 'exposure', 'comparison', 'outcome', 'study_type']:
        try:
            evaluation_dict[f'{key}_evaluation'] = {'decision': zero_shot_result.tool_input[f'{key}_decision'], 'reason': zero_shot_result.tool_input[f'{key}_reason']}

            if evaluation_dict[f'{key}_evaluation']['decision'] not in ['True', 'False', 'NotEvaluable']:
                print(f"Key {key} has invalid decision value: '{evaluation_dict[f'{key}_evaluation']['decision']}'. Inserting NotEvaluable.")
                evaluation_dict[f'{key}_evaluation'] = {'decision': 'NotEvaluable', 'reason': 'LLM provided invalid decision value.'}

        except Exception as e:
            print(f"Key {key} not properly formatted. Assuming {key}_reason was not formatted properly.")
            try:
                evaluation_dict[f'{key}_evaluation'] = {'decision': zero_shot_result.tool_input[f'{key}_decision'], 'reason': 'LLM failed to parse response.'}
            except Exception as e:
                print(print(f"Key {key}_decision not properly formatted. Inserting NotEvaluable."))
                evaluation_dict[f'{key}_evaluation'] = {'decision': 'NotEvaluable', 'reason': 'LLM failed to parse repsonse.'}

    summary = {"True": 0, "False": 0, "NotEvaluable": 0}
    
    # Extract evaluation results for each key
    for key in evaluation_dict.keys():
        evaluation = evaluation_dict[key]
        # Count occurrences of each evaluation result
        try:
            summary[evaluation["decision"]] += 1
        except Exception as e:
            print("No evaluation for this key", key, e)
            summary['NotEvaluable'] += 1

    return {
        "evaluations": evaluation_dict,
        "summary": summary,
        'PECOS' : {}
    }

def check_results(batch_results):
    """
    This function checks the results of a batch job for blank entries.
    It counts the number of blank results and prints their indices.
    :param batch_results: A list of results from a batch job.
    """
    # Check for blank results in the batch_results
    blanks = 0
    indices = []
    for i,  entry in enumerate(batch_results):
        if entry:
            print(i, " ", entry['summary'])  
            if entry['summary']['NotEvaluable'] == 5:
                print("HERE _______________________________")
                blanks += 1
                indices.append(i)
        else:
            print(i, " has not been processed.")
    print(f"{blanks} blank results. Affecting indices: {indices} .")

def save_batch(batch, name_output, savedir_pickle):
    ''' Save batch results to pickle files in the specified directory '''
    print("Saving batch :", name_output)
    with open(f'{savedir_pickle}/{name_output}.pkl', 'wb') as f:
        pickle.dump(batch, f)


def read_ris(file_path):
    """
    Reads a .ris file and converts it to a pandas DataFrame.
    Handles multi-line values (continuations) even when continuation lines are not indented,
    by treating any non-empty line that does NOT match a tag line as a continuation of
    the previous tag's value.
    """
    # RIS tag pattern: exactly two alphanumeric chars, two spaces, hyphen, space
    tag_line_re = re.compile(r'^([A-Za-z0-9]{2})\s{2}-\s(.*)$')

    entries = []
    current_entry = {}
    unique_tags = set()
    last_tag = None

    with open(file_path, 'r', encoding='utf-8') as fh:
        for raw_line in fh:
            # Keep leading spaces (not needed for detection here, but harmless).
            # Trim only newline characters.
            line = raw_line.rstrip('\r\n')

            # Skip completely blank lines
            if not line.strip():
                continue

            # End of record
            if line.startswith('ER  -'):
                if current_entry:
                    entries.append(current_entry)
                current_entry = {}
                last_tag = None
                continue

            m = tag_line_re.match(line)
            if m:
                tag, value = m.group(1), m.group(2)
                unique_tags.add(tag)

                if tag in current_entry:
                    if isinstance(current_entry[tag], list):
                        current_entry[tag].append(value)
                    else:
                        current_entry[tag] = [current_entry[tag], value]
                else:
                    current_entry[tag] = value
                last_tag = tag
            else:
                # Continuation line (no tag prefix)
                if last_tag is not None:
                    cont = line.strip()
                    if cont:
                        if isinstance(current_entry[last_tag], list):
                            current_entry[last_tag][-1] += " " + cont
                        else:
                            current_entry[last_tag] += " " + cont
                # If no last_tag, stray line—ignore or collect as needed.

        # Handle last entry if file doesn't end with 'ER  -'
        if current_entry:
            entries.append(current_entry)

    df = pd.DataFrame(entries)

    # Ensure all encountered tags exist as columns
    for tag in unique_tags:
        if tag not in df.columns:
            df[tag] = None

    # Optional: collapse single-element lists to scalars
    # If you want AU/KW to always be lists, remove or customize this step.
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

    return df


def write_ris(df, filename):
    # writes dataframe to a .ris format
    df.fillna('', inplace =True)  # replace NaN by empty strings to not cause problems downstream
    with open(filename, 'w', encoding='utf-8') as file:
        for index, row in df.iterrows():
            for col in df.columns:
                if row[col] == '':
                    continue
                value = row[col]
                if isinstance(value, list):
                    # Process a list possibly containing sub-lists
                    for item in value:
                        if isinstance(item, list):
                            # Process each sub-item in the second-level list to a string
                            for sub_item in item:
                                file.write(f"{col}  - {sub_item}\n")
                        else:
                            # Write each first-level item directly
                            file.write(f"{col}  - {item}\n")
                else:
                    # Write non-list items directly
                    file.write(f"{col}  - {value}\n")
            file.write("ER  - \n\n")  # denote end of a record for .ris format 