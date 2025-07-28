
import ast, argparse, pandas as pd, torch, warnings
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, IdeficsForVisionText2Text

warnings.filterwarnings("ignore", category=UserWarning)         # silent HF msgs

# ---------- helpers ---------------------------------------------------

def flatten_msgs(raw):
    """
    Accept strings, lists, NaN – return a flat list[dict].
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []

    if isinstance(raw, str):
        raw = ast.literal_eval(raw.replace("nan", "''"))

    out = []

    def _unpack(x):
        if isinstance(x, list):
            for y in x: _unpack(y)
        elif isinstance(x, dict):
            out.append(x)

    _unpack(raw)
    return out


def stringify(msg_dicts):
    """
    msg_dicts: iterable of {'role': .., 'text': ..}
    returns single newline‑separated string.
    """
    parts = []
    for d in msg_dicts:
        role = d.get("role", "unknown")
        text = d.get("text", "")
        parts.append(f"{role}: {text}")
    return "\n".join(parts)


# ---------- main -------------------------------------------------------

def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16

    processor = AutoProcessor.from_pretrained(opt.model)
    model = IdeficsForVisionText2Text.from_pretrained(
        opt.model, device_map="auto", torch_dtype=dtype
    ).eval()

    df   = pd.read_csv(opt.data_path)
    grid = Image.open(opt.image_path).convert("RGB")

    sys_prompt = (
        "You are shown a conversation between a describer and matcher trying to identify an image "
        "among labeled options (A–L). Based on the conversation and the image, guess which tangram "
        "is being described.\nAnswer with a single capital letter from A to L. Do not explain."
    )

    rows_out = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="trials"):
        try:
            raw = (
                row["message_history_trunc"] if opt.history_type == "yoked"
                else row["messages"]
            )
            conv_txt = stringify(flatten_msgs(raw)).strip()
            if not conv_txt:
                raise ValueError("empty conv")

            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": conv_txt},
                {"role": "user",   "content": grid},
            ]

            inp = processor(msgs, return_tensors="pt").to(device, dtype)
            with torch.no_grad():
                pred_ids = model.generate(**inp, max_new_tokens=5, do_sample=False)
            pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

            rows_out.append({"trial_id": row["trial_id"], "model_choice": pred})

        except Exception as e:
            print(f"trial {idx} (id={row.get('trial_id', '?')}): {e}")

    out_csv = (
        f"model_choices-{opt.model.replace('/','--')}-{opt.experiment_name}"
        f"-idefics-{opt.history_type}.csv"
    )
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"saved → {out_csv}   ({len(rows_out)} rows)")

# ---------- CLI --------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HuggingFaceM4/idefics-9b-instruct")
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--history_type", choices=["yoked", "none"], default="yoked")
    p.add_argument("--data_path",  default="trials_with_history.csv")
    p.add_argument("--image_path", default="compiled_grid.png")
    main(p.parse_args())
