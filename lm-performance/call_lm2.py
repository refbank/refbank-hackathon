import argparse, pandas as pd, torch, re
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, IdeficsForVisionText2Text

SYS = (
    "You are shown a conversation between a describer and matcher trying to identify an image among "
    "labeled options (A to L). Based on the conversation and the image, guess which tangram (labeled "
    "A to L) is being described. Answer with a single capital letter from A to L. Do not explain.\n\n"
)

def clean(txt: str) -> str:
    """Return a plain conversation string (strip brackets / quotes that slipped through)."""
    txt = str(txt)
    if txt.lower() == "nan":
        return ""
    # strip leading/trailing list brackets if present
    if txt.lstrip().startswith("[") and txt.rstrip().endswith("]"):
        txt = txt.strip()[1:-1]
    # collapse multiple spaces
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def main(opt):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoProcessor.from_pretrained(opt.model)
    model = IdeficsForVisionText2Text.from_pretrained(
        opt.model, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    df   = pd.read_csv(opt.data_path)
    img  = Image.open(opt.image_path).convert("RGB")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="trials"):
        conv = clean(row["message_history_trunc"] if opt.history_type=="yoked"
                     else row.get("utterance",""))
        if not conv:
            print(f"trial {row['trial_id']}: empty conv"); continue

        prompt = SYS + conv       # ONE single string
        # ONE image (same for every trial)
        inputs = proc(text=[prompt], images=[img], return_tensors="pt").to(dev, torch.float16)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        ans = proc.batch_decode(out, skip_special_tokens=True)[0].strip()
        rows.append({"trial_id": row["trial_id"], "model_choice": ans})

    out_f = f"model_choices-{opt.model.replace('/','--')}-{opt.experiment_name}-idefics-{opt.history_type}.csv"
    pd.DataFrame(rows).to_csv(out_f, index=False)
    print("saved →", out_f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--history_type", choices=["yoked","none"], default="yoked")
    p.add_argument("--data_path",  default="trials_with_history.csv")
    p.add_argument("--image_path", default="compiled_grid.png")
    main(p.parse_args())
