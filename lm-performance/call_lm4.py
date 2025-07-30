# call_lm4.py
import argparse, json, re, os, torch, pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import LlavaProcessor, LlavaForConditionalGeneration

SYSTEM_PROMPT = (
    "You are shown a conversation between a describer and matcher that refers "
    "to an image grid of tangrams labelled A–L. "
    "Based on the conversation **and** the image, guess which letter the "
    "describer is targeting.\n"
    "Answer with a **single capital letter (A–L) only**, no explanation."
)

LETTER_RE = re.compile(r"\b([A-L])\b")

def extract_letter(txt: str) -> str:
    m = LETTER_RE.search(txt.upper())
    return m.group(1) if m else "?"

def load_conv(row, history_type="yoked"):
    if history_type == "yoked":
        blob = row["message_history_trunc"]
        if isinstance(blob, str) and blob.strip():
            return json.loads(blob.replace("''", '""'))[-1]  # last exchange
    # fallback: single‑round utterance
    return json.loads(row["messages"])[-1]["text"]

def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = LlavaProcessor.from_pretrained(opt.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        opt.model,
        torch_dtype=torch.float16,
        device_map="auto"          # loads to GPU, off‑loads if not enough RAM
    ).eval()

    df = pd.read_csv(opt.data_path)
    grid_img = Image.open(opt.image_path).convert("RGB")

    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="trials"):
        try:
            conv = load_conv(row, opt.history_type)
            if not conv:                             # empty conversation
                raise ValueError("empty conv")

            prompt = f"{SYSTEM_PROMPT}\n\n{conv}\n\nAnswer:"
            # processor expects **lists** of text / images
            inputs = processor(
                text=[prompt],
                images=[grid_img],
                return_tensors="pt"
            ).to(device, torch.float16)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.0
                )

            answer = extract_letter(
                processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            )
        except Exception as err:
            print(f"trial {row['trial_id']}: {err}")
            answer = "?"

        preds.append(answer)

    df["model_choice"] = preds
    out_file = (
        f"model_choices-{opt.model.replace('/','--')}-"
        f"{opt.experiment_name}-llava-{opt.history_type}.csv"
    )
    df[["trial_id", "model_choice", "target"]].to_csv(out_file, index=False)
    print(f"saved → {out_file}   ({len(df)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.6-7b-hf")
    ap.add_argument("--experiment_name", required=True)
    ap.add_argument("--history_type", choices=["yoked", "none"], default="yoked")
    ap.add_argument("--data_path", default="trials_with_history.csv")
    ap.add_argument("--image_path", default="compiled_grid.png")
    opt = ap.parse_args()
    main(opt)
