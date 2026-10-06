import pandas as pd

model_list = ["gpt-5.2", "gpt-5", "gpt-5-nano", "gpt-4.1-nano", "o3-mini"]

for model in model_list:
    #load datasets
    llm_zero_df = pd.read_json(f"{model}_zero_shot.jsonl", lines=True)
    llm_persona_df = pd.read_json(f"{model}_persona.jsonl", lines=True)
    llm_fs_df = pd.read_json(f"{model}_few_shot.jsonl", lines=True)
    benchmark_df = pd.read_json("summ_eval_final.json", lines=True) 

    #get rid of raw response, prompt type, and prompt
    llm_zero_df = llm_zero_df.drop(columns=["raw_response", "prompt_type", "prompt"])
    llm_persona_df = llm_persona_df.drop(columns=["raw_response", "prompt_type", "prompt"])
    llm_fs_df = llm_fs_df.drop(columns=["raw_response", "prompt_type", "prompt"])

    #get rid of decoded and text in benchmark
    benchmark_df = benchmark_df.drop(columns=["decoded", "text"])

    #flatten benchmark rouge and ratings
    benchmark_flat = pd.json_normalize(
        benchmark_df.to_dict(orient='records'),
        sep='_'
    )

    #rename
    benchmark_flat = benchmark_flat.rename(columns={
        "rouge_rouge1": "ROUGE-1",
        "rouge_rouge2": "ROUGE-2",
        "rouge_rougeL": "ROUGE-L",
        "bertscore_bert_f1": "BERTScore",
        "turker_avg_scores_coherence": "Crowd_Coherence",
        "turker_avg_scores_consistency": "Crowd_Consistency",
        "turker_avg_scores_fluency": "Crowd_Fluency",
        "turker_avg_scores_relevance": "Crowd_Relevance",
        "expert_avg_scores_coherence": "Expert_Coherence",
        "expert_avg_scores_consistency": "Expert_Consistency",
        "expert_avg_scores_fluency": "Expert_Fluency",
        "expert_avg_scores_relevance": "Expert_Relevance",
    })


    #standardize columns
    for df in [llm_zero_df, llm_persona_df, benchmark_df]:
        df['id'] = df['id'].astype(str)
        df['model_id'] = df['model_id'].astype(str)

    #rename rating cols
    llm_zero_df = llm_zero_df.rename(columns={"fluency": "zs_fluency", "coherence": "zs_coherence", "consistency": "zs_consistency", "relevance": "zs_relevance"})
    llm_persona_df = llm_persona_df.rename(columns={"fluency": "pers_fluency", "coherence": "pers_coherence", "consistency": "pers_consistency", "relevance": "pers_relevance"})
    llm_fs_df = llm_fs_df.rename(columns={"fluency": "fs_fluency", "coherence": "fs_coherence", "consistency": "fs_consistency", "relevance": "fs_relevance"})

    #merge
    merged_df = (
        benchmark_flat
        .merge(llm_zero_df[["id", "model_id", "zs_fluency", "zs_coherence", "zs_consistency", "zs_relevance"]], on=["id", "model_id"], how="inner")
        .merge(llm_persona_df[["id", "model_id", "pers_fluency", "pers_coherence", "pers_consistency", "pers_relevance"]], on=["id", "model_id"], how="inner")
        .merge(llm_fs_df[["id", "model_id", "fs_fluency", "fs_coherence", "fs_consistency", "fs_relevance"]], on=["id", "model_id"], how="inner")
    )

    print(merged_df.shape)
    print(merged_df.columns)
    print(merged_df.head())

    #save
    merged_df.to_json(f"{model}_merged_summ_eval.jsonl", lines=True, orient="records")