import pandas as pd
import argparse
from prompt_for_DocVQA import work
from datasets import load_dataset
import os
from PIL import Image

def prepare_docvqa_data(num_shot, num_test_samples=10):
    """Prepare DocVQA data from Hugging Face datasets"""
    dataset = load_dataset("nielsr/docvqa_1200_examples")
    
    # Create image directory
    image_folder = "./dataset/DocVQA/images"
    os.makedirs(image_folder, exist_ok=True)
    
    def process_split(split, num_samples):
        samples = []
        for example in split.select(range(num_samples)):
            # Save image
            img_path = os.path.join(image_folder, f"{example['id']}.png")
            example['image'].save(img_path)
            
            samples.append({
                'image_id': example['id'],
                'question': example['query'],
                'answer': example['answer']
            })
        return pd.DataFrame(samples)
    
    # Prepare demo and test data
    demo_df = process_split(dataset['train'], num_shot)
    test_df = process_split(dataset['test'], num_test_samples)
    
    return demo_df, test_df, image_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocVQA Experiment")
    parser.add_argument(
        "--num_shot",
        type=int,
        required=True,
        help="Number of demonstration examples"
    )
    parser.add_argument(
        "--num_qns_per_round",
        type=int,
        default=1,
        help="Questions per API call"
    )

    args = parser.parse_args()

    args.model = "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4"

    # Prepare DocVQA data
    demo_df, test_df, image_folder = prepare_docvqa_data(args.num_shot)
    
    print(f"Demo examples: {len(demo_df)}")
    print(f"Test examples: {len(test_df)}")
    print(f"Images saved to: {image_folder}")

    # Run the experiment
    work(
        model=args.model,
        num_shot=args.num_shot,
        location='usa',
        num_qns_per_round=args.num_qns_per_round,
        test_df=test_df,
        demo_df=demo_df,
        SAVE_FOLDER=image_folder,
        dataset_name="DocVQA",
        file_suffix=".png"
    )