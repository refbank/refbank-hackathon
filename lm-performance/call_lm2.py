import argparse
import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, IdeficsForVisionText2Text
from tqdm import tqdm

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model)
    model = IdeficsForVisionText2Text.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    df = pd.read_csv(args.data_path)
    image = Image.open(args.image_path).convert("RGB")

    system_prompt = (
        "You are shown a conversation between a describer and matcher trying to identify an image among labeled options (A to L). "
        "Based on the conversation and the image, guess which tangram (labeled A to L) is being described.\n"
        "Answer with a single capital letter from A to L. Do not include any explanation."
    )

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        conv = row["message_history_trunc"] if args.history_type == "yoked" else row.get("utterance", "")

        if not isinstance(conv, str):
            conv = str(conv)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [conv.strip(), image]},
        ]

        try:
            inputs = processor(messages, return_tensors="pt")
            inputs = {k: v.to(device, dtype=torch.float16) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False
                )

            decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        except Exception as e:
            print(f"Error on trial_id={row.get('trial_id', 'UNKNOWN')}: {e}")
            decoded = "ERROR"

        results.append({
            "trial_id": row.get("trial_id", ""),
            "model_choice": decoded
        })

    out_path = f"model_choices-{args.model.replace('/', '--')}-{args.experiment_name}-idefics-history-{args.history_type}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--history_type", choices=["yoked", "none"], default="yoked")
    parser.add_argument("--data_path", default="trials_with_history.csv")
    parser.add_argument("--image_path", default="compiled_grid.png")
    args = parser.parse_args()
    main(args)
