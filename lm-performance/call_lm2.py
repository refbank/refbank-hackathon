import argparse, ast, warnings, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, IdeficsForVisionText2Text

warnings.filterwarnings("ignore", category=UserWarning)   # silence HF chatter

# ---------- helpers ----------------------------------------------------
def _flatten(obj):
    "Recursively collect dicts inside lists/strings → list[dict]"
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return []
    
    if isinstance(obj, str):
        # Handle empty string cases
        if not obj.strip() or obj.strip() == '[]':
            return []
        try:
            # Try to parse the string representation
            obj = ast.literal_eval(obj.replace("nan", "''"))
        except (ValueError, SyntaxError) as e:
            print(f"Failed to parse string with ast.literal_eval: {e}")
            return []
    
    # Handle the case where obj is already a list
    if isinstance(obj, list) and not obj:
        return []
    
    out = []
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, dict):
            out.append(x)
    return out[::-1]   # keep original order

def _to_chat_str(msg_dicts):
    return "\n".join(f"{d.get('role','?')}: {d.get('text','')}" for d in msg_dicts)

# ---------- main -------------------------------------------------------
def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16

    proc  = AutoProcessor.from_pretrained(opt.model)
    model = IdeficsForVisionText2Text.from_pretrained(
        opt.model, device_map="auto", torch_dtype=dtype
    ).eval()

    df   = pd.read_csv(opt.data_path)
    grid = Image.open(opt.image_path).convert("RGB")

    system_prompt = (
        "You are shown a conversation between a describer and matcher trying to identify an image "
        "among labeled options (A to L).\n"
        "Based on the conversation and the image, guess which tangram (labeled A to L) is being described.\n"
        "Answer with a single capital letter from A to L. Do not explain."
    )

    rows_out = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="trials"):
        try:
            raw = row["message_history_trunc"] if opt.history_type == "yoked" \
                  else row["messages"]
            txt = _to_chat_str(_flatten(raw)).strip()
            if not txt:
                raise ValueError("empty conv")

            prompt = f"{system_prompt}\n\n{txt}"
            inputs = proc(images=grid, text=prompt, return_tensors="pt")
            
            # Fixed: Only move to device, don't change dtype for all tensors
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)
            
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            pred = proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

            rows_out.append({"trial_id": row["trial_id"], "model_choice": pred})

        except Exception as e:
            print(f"trial {idx} (id={row.get('trial_id', '?')}): {e}")

    out_csv = (
        f"model_choices-{opt.model.replace('/','--')}-"
        f"{opt.experiment_name}-idefics-{opt.history_type}.csv"
    )
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"saved → {out_csv}   ({len(rows_out)} rows)")

# ---------- CLI --------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--history_type", choices=["yoked", "none"], default="yoked")
    p.add_argument("--data_path",  default="trials_with_history.csv")
    p.add_argument("--image_path", default="compiled_grid.png")
    main(p.parse_args())