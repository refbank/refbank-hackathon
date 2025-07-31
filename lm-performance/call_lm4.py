import argparse, ast, warnings, re, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings("ignore", category=UserWarning)
LETTER_RE = re.compile(r"\b([A-L])\b")

def _flatten(obj):
    """Recursively collect dicts inside lists/strings → list[dict]"""
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return []
    if isinstance(obj, str):
        if not obj.strip() or obj.strip() == "[]":
            return []
        try:
            obj = ast.literal_eval(obj.replace("nan", "''"))
        except (ValueError, SyntaxError):
            return []
    if isinstance(obj, list) and not obj:
        return []
    out, stack = [], [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, dict):
            out.append(x)
    return out[::-1]

def _to_chat_str(dicts):
    return "\n".join(f"{d.get('role', '?')}: {d.get('text','')}" for d in dicts)

def extract_letter(text: str) -> str:
    m = LETTER_RE.search(text.upper().strip())
    return m.group(1) if m else "?"

def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Try basic CLIP (should work with existing transformers)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    except Exception as e:
        print(f"Failed to load CLIP: {e}")
        print("Try: pip cache purge && pip install transformers --upgrade")
        return

    df = pd.read_csv(opt.data_path)
    print(f"Original dataset has {len(df)} rows")
    df = df.head(300)
    print(f"Testing with {len(df)} rows")

    grid_img = Image.open(opt.image_path).convert("RGB")

    rows_out = []
    successful_predictions = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="trials"):
        try:
            raw = row["message_history_trunc"] if opt.history_type == "yoked" else row["messages"]
            conv_text = _to_chat_str(_flatten(raw)).strip()
            if not conv_text:
                raise ValueError("empty conv")

            # Create candidate descriptions - randomize order to avoid bias
            import random
            letters = list("ABCDEFGHIJKL")
            random.shuffle(letters)  # Randomize order each time
            
            candidates = []
            for letter in letters:
                # Try different prompt styles
                candidate = f"This describes tangram piece {letter}: {conv_text[:150]}"
                candidates.append(candidate)
            
            # Process image and text
            inputs = processor(
                text=candidates,
                images=grid_img,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
                # Add temperature sampling to reduce bias
                temperature = 2.0  # Higher temperature = more random
                adjusted_logits = logits_per_image / temperature
                adjusted_probs = adjusted_logits.softmax(dim=1)
                
                # Sample from the distribution instead of taking argmax
                best_idx = torch.multinomial(adjusted_probs, 1).item()
                pred_letter = letters[best_idx]  # Map back to original letter
                confidence = probs[0, best_idx].item()

            rows_out.append({
                "trial_id": row["trial_id"],
                "model_choice_raw": f"Letter {pred_letter} (confidence: {confidence:.3f})",
                "model_choice": pred_letter,
                "target": row["target"],
                "correct": pred_letter == row["target"],
                "valid_format": True
            })

            successful_predictions += 1

            if idx < 6:
                print(f"\nTrial {idx} id={row['trial_id']}  tgt={row['target']}  pred={pred_letter} conf={confidence:.3f}")

        except Exception as err:
            print(f"trial {idx} (id={row.get('trial_id','?')}): {err}")
            rows_out.append({
                "trial_id": row.get("trial_id", f"unknown_{idx}"),
                "model_choice_raw": f"ERROR: {str(err)}",
                "model_choice": "?",
                "target": row.get("target", "?"),
                "correct": False,
                "valid_format": False
            })

    # Save results
    out_csv = f"model_choices-clip-{opt.experiment_name}-{opt.history_type}.csv"
    
    if not rows_out:
        print("No predictions were made!")
        return
        
    res_df = pd.DataFrame(rows_out)
    res_df.to_csv(out_csv, index=False)

    total = len(res_df)
    correct = res_df["correct"].sum()
    valid = res_df["valid_format"].sum()
    
    print(f"\nSaved → {out_csv} ({total} rows)")
    print(f"Successful predictions: {successful_predictions}/{total}")
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    print(f"Valid format: {valid}/{total} = {valid/total*100:.2f}%")
    print(f"Random baseline: {100/12:.2f}%")

    if valid > 0:
        from collections import Counter
        valid_predictions = res_df[res_df["valid_format"]]
        dist = Counter(valid_predictions["model_choice"])
        print("\nPrediction distribution (valid only):")
        for l in "ABCDEFGHIJKL":
            if dist[l] > 0:
                print(f"  {l}: {dist[l]} ({dist[l]/valid*100:.1f}%)")
        
        if valid > 0 and max(dist.values()) > valid * 0.3:
            mc = dist.most_common(1)[0]
            print(f"\n⚠️  Bias warning – '{mc[0]}' appears {mc[1]/valid*100:.1f}% of valid predictions")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--history_type", choices=["yoked", "none"], default="yoked")
    p.add_argument("--data_path", default="trials_with_history.csv")
    p.add_argument("--image_path", default="compiled_grid.png")
    main(p.parse_args())