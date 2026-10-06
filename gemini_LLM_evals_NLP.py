import google.generativeai as genai

import openai
import json
from tqdm import tqdm
import re

#added import below to fix directory issues
from pathlib import Path

#model list
model_list = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

def build_zero_shot_prompt(text, decoded):
    #zero shot template
    zero_template = """
    In this task you will evaluate the quality of a summary written for a news article. 
    Carefully read the news article, be aware of the information it contains.
    Rate each summary on a scale from 1 (worst) to 5 (best) by its relevence, consistency, fluency, and coherence.

    - **Relevance**: The rating measures how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary.
    - **Consistency**: The rating measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.
    - **Fluency**: This rating measures the quality of individual sentences, are they well-written and dramatically correct. Consider the quality of individual sentences.
    - **Coherence**: The rating measures the quality of all sentences collectively, to the fit together and sound naturally. Consider the quality of the summary as a whole.

    Only return your ratings in JSON format like:
    {{
    "fluency": int,
    "coherence": int,
    "consistency": int,
    "relevance": int
    }}

    ARTICLE:
    {text_input}

    SUMMARY:
    {decoded_input}
    """
    return zero_template.format(text_input=text, decoded_input=decoded)


def build_persona_prompt(text, decoded):
    #persona template
    persona_template = """
    You are a professional summarization evaluator working for a prominent newspaper. 

    In this task you will evaluate the quality of a summary written for a news article. 
    Carefully read the news article, be aware of the information it contains.
    Rate each summary on a scale from 1 (worst) to 5 (best) by its relevence, consistency, fluency, and coherence.

    - **Relevance**: The rating measures how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary.
    - **Consistency**: The rating measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.
    - **Fluency**: This rating measures the quality of individual sentences, are they well-written and dramatically correct. Consider the quality of individual sentences.
    - **Coherence**: The rating measures the quality of all sentences collectively, to the fit together and sound naturally. Consider the quality of the summary as a whole.

    Only return your ratings in JSON format like:
    {{
      "fluency": int,
      "coherence": int,
      "consistency": int,
      "relevance": int
    }}

    ARTICLE:
    {text_input}

    SUMMARY:
    {decoded_input}
    """
    return persona_template.format(text_input=text, decoded_input=decoded)

#FILL IN YOUR API KEY HERE
genai.configure(api_key="")

def call_llm(prompt, model):
    response = model.generate_content(prompt)
    return response.text

def extract_json(raw):
    # Remove markdown ```json ... ``` fences if present
    match = re.search(r'```json(.*?)```', raw, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        return json.loads(json_str)
    # Fallback: just look for the first {...} block
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No JSON object found in model output!")

def evaluate_and_save(data, outfile, prompt_type, model):
    # added two lines below to fix directory issues
    outfile_path = Path(outfile)
    outfile_path.parent.mkdir(parents=True, exist_ok=True)
    # Initialize the variable before the try block to avoid UnboundLocalError
    raw = None 
    results = []
    with open(outfile, "w", encoding="utf-8") as f_out:
        for item in tqdm(data):
            # Generate the prompt based on the template
            if prompt_type == "zero-shot":
                prompt = build_zero_shot_prompt(item['text'], item['decoded'])
            else:
                prompt = build_persona_prompt(item['text'], item['decoded'])
            try:
                raw = call_llm(prompt, model)
                parsed = extract_json(raw)
                output = {
                    "id": item["id"],
                    "model_id": item["model_id"],
                    "fluency": parsed["fluency"],
                    "coherence": parsed["coherence"],
                    "consistency": parsed["consistency"],
                    "relevance": parsed["relevance"],
                    "raw_response": raw,
                    "prompt_type": prompt_type,
                    "prompt": prompt
                }
                f_out.write(json.dumps(output) + "\n")
            except Exception as e:
                print(f"Failed for {item['id']} - {e}")
                print("Raw was: ", repr(raw))

#load samples
with open("summ_eval_final.json", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

if __name__ == "__main__":
    for model in model_list:
        model = genai.GenerativeModel(model)
        #run zero shot
        evaluate_and_save(data, f"{model.model_name}_zero_shot.jsonl", "zero-shot", model)

        #run persona
        #evaluate_and_save(data, f"{model.model_name}_persona.jsonl", "persona", model)