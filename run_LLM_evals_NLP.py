import openai
import json
from tqdm import tqdm

#models
model_list = ["gpt-5", "gpt-5-nano", "gpt-4.1-nano", "o3-mini"] #["gpt-5.2"]
               #

#zero shot template
ZERO_SHOT_TEMPLATE = f"""
In this task you will evaluate the quality of a summary written for a news article. 
Carefully read the news article, be aware of the information it contains.
Rate each summary on a scale from 1 (worst) to 5 (best) by its relevence, consistency, fluency, and coherence.

- **Relevance**: The rating measures how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary.
- **Consistency**: The rating measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.
- **Fluency**: This rating measures the quality of individual sentences, are they well-written and dramatically correct. Consider the quality of individual sentences.
- **Coherence**: The rating measures the quality of all sentences collectively, to the fit together and sound naturally. Consider the quality of the summary as a whole.

Only return your ratings in JSON format like:
  "fluency": int,
  "coherence": int,
  "consistency": int,
  "relevance": int

ARTICLE:
{{text}}

SUMMARY:
{{decoded}}
"""


#few shot template
import json

# Load the few-shot examples from the JSONL file
with open("few_shot_example.json", "r", encoding="utf-8") as f:
    few_shot_data = [json.loads(line) for line in f]

# Format each example into a block
def format_few_shot_example(entry):
    scores = entry['expert_avg_scores']
    return f"""
ARTICLE:
{entry['text']}

SUMMARY:
{entry['decoded']}

RATING:
{{
  "fluency": {round(scores['fluency'])},
  "coherence": {round(scores['coherence'])},
  "consistency": {round(scores['consistency'])},
  "relevance": {round(scores['relevance'])}
}}
""".strip()

#combine the examples with separators
few_shot_block = "\n\n### EXAMPLE\n\n".join(
    format_few_shot_example(e) for e in few_shot_data
)

#few shot template
FEW_SHOT_TEMPLATE = f"""
In this task you will evaluate the quality of a summary written for a news article. 
Carefully read the news article, be aware of the information it contains.
Rate each summary on a scale from 1 (worst) to 5 (best) by its relevence, consistency, fluency, and coherence.

- **Relevance**: The rating measures how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary.
- **Consistency**: The rating measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.
- **Fluency**: This rating measures the quality of individual sentences, are they well-written and dramatically correct. Consider the quality of individual sentences.
- **Coherence**: The rating measures the quality of all sentences collectively, to the fit together and sound naturally. Consider the quality of the summary as a whole.

Only return your ratings in JSON format like:
  "fluency": int,
  "coherence": int,
  "consistency": int,
  "relevance": int

Below are examples of summaries and their ratings.

### EXAMPLE

{few_shot_block}

### Now evaluate the following:

ARTICLE:
{{text}}

SUMMARY:
{{decoded}}

RATING:
"""

#persona template
#what do we want to do for this????
PERSONA_TEMPLATE = f"""
You are a professional summarization evaluator working for a prominent newspaper. 

In this task you will evaluate the quality of a summary written for a news article. 
Carefully read the news article, be aware of the information it contains.
Rate each summary on a scale from 1 (worst) to 5 (best) by its relevence, consistency, fluency, and coherence.

- **Relevance**: The rating measures how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary.
- **Consistency**: The rating measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.
- **Fluency**: This rating measures the quality of individual sentences, are they well-written and dramatically correct. Consider the quality of individual sentences.
- **Coherence**: The rating measures the quality of all sentences collectively, to the fit together and sound naturally. Consider the quality of the summary as a whole.


Only return your ratings in JSON format like:
  "fluency": int,
  "coherence": int,
  "consistency": int,
  "relevance": int

ARTICLE:
{{text}}

SUMMARY:
{{decoded}}
"""

API_KEY = ""

def generate_prompt(template, text, decoded):
    return template.format(text=text, decoded=decoded)

def generate_fs_prompt(template: str, text: str, decoded: str) -> str:
    return template.replace("{text}", text).replace("{decoded}", decoded)


def call_gpt(prompt, model):
    client = openai.OpenAI(api_key = API_KEY)
    model_name =  model  #TODO: CHANGE THIS
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def evaluate_and_save(data, template, outfile, prompt_type, model):
    with open(outfile, "w", encoding="utf-8") as f_out:
        for item in tqdm(data):
            if prompt_type == "few-shot":
                prompt = generate_fs_prompt(template, item['text'], item['decoded'])
            else:
                prompt = generate_prompt(template, item['text'], item['decoded'])
            try:
                raw = call_gpt(prompt, model)
                parsed = json.loads(raw)
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

#load samples
with open("summ_eval_final.json", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

if __name__ == "__main__":
    for model in model_list:
        #run zero shot
        #evaluate_and_save(data, ZERO_SHOT_TEMPLATE, f"{model}_zero_shot.jsonl", "zero-shot", model)

        #run few shot
        evaluate_and_save(data, FEW_SHOT_TEMPLATE, f"{model}_few_shot.jsonl", "few-shot", model)

        #run persona
        #evaluate_and_save(data, PERSONA_TEMPLATE, f"{model}_persona.jsonl", "persona", model)