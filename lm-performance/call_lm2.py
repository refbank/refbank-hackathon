import ast
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

from transformers import AutoProcessor, IdeficsForVisionText2Text


# ---------- helpers -------------------------------------------------------

def stringify(turns):
    """
    turns  – list[dict]  OR  string repr of that list  OR  NaN
    returns one readable multi‑line string.
    """
    if turns is None or (isinstance(turns, float) and pd.isna(turns)):
        return ""

    if isinstance(turns, str):
        turns = ast.literal_eval(turns.replace("nan", "''"))

    if not isinstance(turns, list):
        return str(turns)

    lines = []
    for msg in turns:
        role = msg.get("role", "unknown")
        text = msg.get("text", "")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


# ---------- main ----------------------------------------------------------

def main(args):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype    = torch.float16

    processor = AutoProcessor.from_pretrained(args.model)
    model     = IdeficsForVisionText2Text.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype
    ).eval()

    df = pd.read_csv(args.data_path)
    grid_image = Image.open(args.image_path).convert("RGB")

    sys_prompt = (
        "You are shown a conversation between a describer and matcher trying to identify an image among labeled "
        "options (A to L). Based on the conversation and the image, guess which tangram (labeled A to L) is being "
        "described.\nAnswer with a single capital letter from A to L. Do not include any explanation."
    )

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="trials"):
        try:
            if args.history_type == "yoked":
                conv_raw = row["message_history_trunc"]
            else:
                conv_raw = row["messages"]          # single‑turn version

            conv_text = stringify(conv_raw)
            if not conv_text.strip():
                raise ValueError("empty conv")

            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": conv_text},
                {"role": "user",   "content": grid_image},
            ]

            inputs = processor(msgs, return_tensors="pt").to(device, dtype)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=5, do_sample=False)

            pred = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            results.append({"trial_id": row["trial_id"], "model_choice": pred})

        except Exception as e:
            tid = row.get("trial_id", "unknown")
            print(f"trial {tid}: {e}")

    # save -----------------------------------------------------------------
    out_name = (
        f"model_choices-{args.model.replace('/','--')}-"
        f"{args.experiment_name}-idefics-{args.history_type}.csv"
    )
    pd.DataFrame(results).to_csv(out_name, index=False)
    print(f"saved → {out_name}")


# ---------- cli -----------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="e.g. HuggingFaceM4/idefics-9b-instruct")
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--history_type", choices=["yoked", "none"],
                   default="yoked")
    p.add_argument("--data_path",  default="trials_with_history.csv")
    p.add_argument("--image_path", default="compiled_grid.png")
    args = p.parse_args()
    main(args)
