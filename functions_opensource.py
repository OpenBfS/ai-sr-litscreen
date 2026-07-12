from pydantic import BaseModel, Field, field_validator
from typing import Literal
import json
from langchain.schema import AIMessage
import re
import pandas as pd


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


class ValidateForInclusionLight(BaseModel):
    population: Literal['True', 'False', 'NotEvaluable'] = Field(
        description="Evaluation result of the study population: True (include), False (exclude), or NotEvaluable"
    )

    exposure: Literal['True', 'False', 'NotEvaluable'] = Field(
        description="Evaluation result of the exposure: True (include), False (exclude), or NotEvaluable"
    )

    comparison: Literal['True', 'False', 'NotEvaluable'] = Field(
        description="Evaluation result of the comparison: True (include), False (exclude), or NotEvaluable"
    )

    outcome: Literal['True', 'False', 'NotEvaluable'] = Field(
        description="Evaluation result of the outcome: True (include), False (exclude), or NotEvaluable"
    )

    study_type: Literal['True', 'False', 'NotEvaluable'] = Field(
        description="Evaluation result of the study type: True (include), False (exclude), or NotEvaluable"
    )


    @field_validator('population', 'exposure', 'comparison', 'outcome', 'study_type')
    def validate(cls, v):
        if v not in ['True', 'False', 'NotEvaluable']:
            raise ValueError("The decision response must contain either 'True', 'False' or 'NotEvaluable'. No other options are permitted.")
        return v
    

def get_message_zeroshot_noreason(input):

    """
    Need a parse method for the PECOS validation LLM output.
    We are going to need this wrapper to simplify the AIMessage style of the Ollama models and make them similar to OpenAI model output formats.
    :param input: LLM response message
    :return: AIMessage with properly formatted function_call arguments
    """
    blank_msg = AIMessage(content=input.content, additional_kwargs={'function_call' : {'name': 
                                                                'ValidateForInclusionLight', 'arguments' : 
                                                                json.dumps(
                                                                    {'population': 'NotEvaluable',
                                                                    'exposure': 'NotEvaluable',
                                                                    'comparison': 'NotEvaluable',
                                                                     'outcome': 'NotEvaluable',
                                                                    'study_type': 'NotEvaluable'}
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
        for key in ['population', 'exposure', 'comparison','outcome', 'study_type']:
            if key not in arguments:
                print("Model failed to provide PECOS element in output: ", key)
                print(arguments)
                arguments[key] = 'NotEvaluable'
                print("Inserted NE value for: ", key)
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
                    arguments =  {'population': 'NotEvaluable',
                                                                    'exposure': 'NotEvaluable',
                                                                    'comparison': 'NotEvaluable',
                                                                     'outcome': 'NotEvaluable',
                                                                    'study_type': 'NotEvaluable'}
                    # check if the model provided all elements already:
                    for key in ['population', 'exposure', 'comparison','outcome', 'study_type']:
                        if key in rescue_arguments.keys():
                            arguments[key] = rescue_arguments[key] # update arguments
                    msg = AIMessage(content=input.content, additional_kwargs={'function_call' : {'name':
                                                                                        'ValidateForInclusionLight', 'arguments' : json.dumps(arguments)
                                                                                        }
                                                                                        }
                                                                                        )
                    return msg
            print(rescue_format)
            arguments =                                                                     {'population': 'NotEvaluable',
                                                                    'exposure': 'NotEvaluable',
                                                                    'comparison': 'NotEvaluable',
                                                                     'outcome': 'NotEvaluable',
                                                                    'study_type': 'NotEvaluable'}
            # check if the model provided all elements already:
            for key in ['population', 'exposure', 'comparison','outcome', 'study_type']:
                if key in rescue_format.keys():
                    arguments[key] = rescue_format[key] # update arguments
            msg = AIMessage(content=input.content, additional_kwargs={'function_call' : {'name':
                                                                                'ValidateForInclusionLight', 'arguments' : json.dumps(arguments)
                                                                                }
                                                                                }
                                                                                )
            return msg
        
        except json.JSONDecodeError as e:
            # Log the error for debugging
            print("JSON Decode Error:", e)
            
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
                            arguments =  {'population': 'NotEvaluable',
                                                                            'exposure': 'NotEvaluable',
                                                                            'comparison': 'NotEvaluable',
                                                                            'outcome': 'NotEvaluable',
                                                                            'study_type': 'NotEvaluable'}
                            # check if the model provided all elements already:
                            for key in ['population', 'exposure', 'comparison','outcome', 'study_type']:
                                if key in rescue_arguments.keys():
                                    arguments[key] = rescue_arguments[key] # update arguments
                            msg = AIMessage(content=input.content, additional_kwargs={'function_call' : {'name':
                                                                                                'ValidateForInclusionLight', 'arguments' : json.dumps(arguments)
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
            print("Model did not provide the format in the content window. Equating to null response. ", e)
            return blank_msg
    

def format_zeroshot_noreason_results(zero_shot_result):
    # Combine all evaluation results into a single dictionary
    evaluation_dict = {}

    for key in ['population', 'exposure', 'comparison', 'outcome', 'study_type']:
        try:
            evaluation_dict[f'{key}_evaluation'] = {'decision': zero_shot_result.tool_input[f'{key}'], 'reason': 'No reason requested.'}
            if evaluation_dict[f'{key}_evaluation']['decision'] not in ['True', 'False', 'NotEvaluable']:
                print(f"Key {key} has invalid decision value: '{evaluation_dict[f'{key}_evaluation']['decision']}'. Inserting NotEvaluable.")
                evaluation_dict[f'{key}_evaluation'] = {'decision': 'NotEvaluable', 'reason': 'LLM provided invalid decision value.'}
        except Exception as e:
                print(print(f"Key {key} not properly formatted. Inserting NotEvaluable."))
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

def save_batch(batch, name_output, savedir_pickle):
    ''' Save batch results to pickle files in the specified directory '''
    import pickle
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