import argparse
import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HuggingFaceM4/idefics-9b-instruct")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    return parser.parse_args()

def main(args):
    # Load model and processor
    processor = AutoProcessor.from_pretrained(args.model)
    model = IdeficsForVisionText2Text.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # Load image
    image = Image.open(args.image_path).convert("RGB")

    # Load CSV
    df = pd.read_csv(args.input_csv)

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row["prompt"]

        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)

        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results.append({
            "prompt": prompt,
            "prediction": response.strip()
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")

if __name__ == "__main__":
    main(parse_args())
