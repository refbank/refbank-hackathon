# call_idefics.py
import argparse, pandas as pd, torch, re
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, IdeficsForVisionText2Text


SYSTEM_PROMPT = (
    "You are shown a conversation between a describer and matcher trying to identify an image among "
    "labeled options (A to L). Based on the conversation and the image, guess which tangram (labeled "
    "A to L) is being described. Answer with a single capital letter from A to L. Do not explain."
)

def tidy_conv(raw: str) -> str:
    """Remove leading/trailing brackets if the CSV cell still looks like a Python list"""
    if raw.lstrip().startswith("[") and raw.rstrip().endswith("]"):
        # crude fallback – strip outer brackets
        raw = raw.strip()[1:-1]
    return raw.strip()

def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(opt.model)
    model      = IdeficsForVisionText2Text.from_pretrained(
        opt.model, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    df     = pd.read_csv(opt.data_path)
    img    = Image.open(opt.image_path).convert("RGB")

    preds  = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="trials"):
        try:
            conv = row["message_history_trunc"] if opt.history_type == "yoked" else row["utterance"]
            conv = tidy_conv(str(conv))

            if not conv or conv.lower() == "nan":
                raise ValueError("empty conv")

            # Processor wants *separate* text & image lists, same order
            inputs = processor(
                text   =[SYSTEM_PROMPT, conv],
                images =[None,          img],   # None for the system prompt, image for user message
                return_tensors="pt"
            ).to(device, torch.float16)

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            answer = processor.batch_decode(output, skip_special_tokens=True)[0].strip()

        except Exception as e:
            print(f"trial {row.get('trial_id')}: {e}")
            answer = "ERR"

        preds.append({"trial_id": row["trial_id"], "model_choice": answer})

    out = f"model_choices-{opt.model.replace('/','--')}-{opt.experiment_name}-idefics-{opt.history_type}.csv"
    pd.DataFrame(preds).to_csv(out, index=False)
    print("saved →", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--history_type", choices=["yoked", "none"], default="yoked")
    p.add_argument("--data_path",  default="trials_with_history.csv")
    p.add_argument("--image_path", default="compiled_grid.png")
    opt = p.parse_args()
    main(opt)
