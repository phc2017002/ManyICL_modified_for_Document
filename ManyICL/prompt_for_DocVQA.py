import traceback
import os
from tqdm import tqdm
import random
import pickle
import numpy as np
from LMM_with_qwen2vl import Qwen2VLAPI

def work(
    model,
    num_shot,
    location,
    num_qns_per_round,
    test_df,
    demo_df,
    SAVE_FOLDER,
    dataset_name,
    detail="auto",
    file_suffix="",
):
    """
    Run DocVQA queries using demonstrating examples from demo_df.

    model[str]: model checkpoint to use
    num_shot[int]: number of demonstrating examples
    location[str]: Vertex AI location (not used for some models)
    num_qns_per_round[int]: number of queries per API call
    test_df, demo_df[pd.DataFrame]: dataframes containing 'image_id', 'question', and 'answer'
    SAVE_FOLDER[str]: path to images
    dataset_name[str]: name of the dataset
    detail[str]: image resolution detail
    file_suffix[str]: suffix for image filenames
    """

    EXP_NAME = f"{dataset_name}_DocVQA"
    api = Qwen2VLAPI(model=model, detail=detail)

    # Prepare demonstrating examples
    demo_examples = []
    demo_samples = demo_df.sample(n=num_shot, random_state=42)
    for _, row in demo_samples.iterrows():
        demo_examples.append((row['image_id'], row['question'], row['answer']))

    # Load existing results
    if os.path.isfile(f"{EXP_NAME}.pkl"):
        with open(f"{EXP_NAME}.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    test_df = test_df.sample(frac=1, random_state=66)
    for start_idx in tqdm(range(0, len(test_df), num_qns_per_round), desc=EXP_NAME):
        end_idx = min(len(test_df), start_idx + num_qns_per_round)
        
        # Build prompt with examples
        prompt = ""
        image_paths = []
        
        # Add demonstration examples
        for demo in demo_examples:
            img_id, question, answer = demo
            img_path = os.path.join(SAVE_FOLDER, img_id + file_suffix)
            image_paths.append(img_path)
            prompt += f"""<<IMG>>{img_path}
Example Question: {question}
Example Answer: {answer}

"""

        # Add test questions
        test_batch = test_df.iloc[start_idx:end_idx]
        qns_info = []
        for idx, row in enumerate(test_batch.itertuples()):
            qn_idx = idx + 1
            img_path = os.path.join(SAVE_FOLDER, row.image_id + file_suffix)
            image_paths.append(img_path)
            prompt += f"""<<IMG>>{img_path}
Question {qn_idx}: {row.question}
"""
            qns_info.append((row.Index, row.question))

        # Add response format instructions
        prompt += "\nAnswer all questions using this format:\n"
        for qn in range(1, len(test_batch)+1):
            prompt += f"Answer {qn}: [Your answer here]\n"
        prompt += "Do not include any additional text."

        # Process API call
        qns_id = str([qi[0] for qi in qns_info])
        for retry in range(3):
            if qns_id in results and "ERROR" not in results.get(qns_id, ""):
                continue

            try:
                response = api(
                    prompt,
                    image_paths=image_paths,
                    real_call=True,
                    max_tokens=10000*num_qns_per_round  # Increased for text answers
                )
                results[qns_id] = response
            except Exception as e:
                results[qns_id] = f"ERROR: {traceback.format_exc()}"

            print('results-----', results)
            # Save after each attempt
            with open(f"{EXP_NAME}.pkl", "wb") as f:
                pickle.dump(results, f)
            break

    # Save final token usage
    results["token_usage"] = api.token_usage
    with open(f"{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(results, f)