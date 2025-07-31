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
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    except Exception as e:
        print(f"Failed to load CLIP: {e}")
        return

    df = pd.read_csv(opt.data_path)
    print(f"Original dataset has {len(df)} rows")
    df = df.head(50)
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

            # Try multiple approaches and average the results
            all_scores = torch.zeros(12).to(device)  # For A-L
            
            # Approach 1: Direct matching
            candidates1 = []
            for letter in "ABCDEFGHIJKL":
                candidate = f"Tangram piece {letter} fits this description: {conv_text[:150]}"
                candidates1.append(candidate)
            
            inputs1 = processor(text=candidates1, images=grid_img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs1 = model(**inputs1)
                scores1 = outputs1.logits_per_image.softmax(dim=1).squeeze()
                all_scores += scores1
            
            # Approach 2: Question format
            candidates2 = []
            for letter in "ABCDEFGHIJKL":
                candidate = f"Question: Which piece matches '{conv_text[:100]}'? Answer: Piece {letter} matches this description."
                candidates2.append(candidate)
            
            inputs2 = processor(text=candidates2, images=grid_img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs2 = model(**inputs2)
                scores2 = outputs2.logits_per_image.softmax(dim=1).squeeze()
                all_scores += scores2
            
            # Approach 3: Negative contrast
            candidates3 = []
            for letter in "ABCDEFGHIJKL":
                candidate = f"The described tangram piece is specifically piece {letter}, not any other piece. Description: {conv_text[:100]}"
                candidates3.append(candidate)
            
            inputs3 = processor(text=candidates3, images=grid_img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs3 = model(**inputs3)
                scores3 = outputs3.logits_per_image.softmax(dim=1).squeeze()
                all_scores += scores3
            
            # Average the scores from all approaches
            avg_scores = all_scores / 3.0
            best_idx = avg_scores.argmax().item()
            pred_letter = "ABCDEFGHIJKL"[best_idx]
            confidence = avg_scores[best_idx].item()

            rows_out.append({
                "trial_id": row["trial_id"],
                "model_choice_raw": f"Letter {pred_letter} (avg confidence: {confidence:.3f})",
                "model_choice": pred_letter,
                "target": row["target"],
                "correct": pred_letter == row["target"],
                "valid_format": True
            })

            successful_predictions += 1

            if idx < 6:
                print(f"\nTrial {idx} id={row['trial_id']}  tgt={row['target']}  pred={pred_letter} conf={confidence:.3f}")
                # Show top 3 predictions for debugging
                top3_indices = avg_scores.argsort(descending=True)[:3]
                top3_letters = ["ABCDEFGHIJKL"[i] for i in top3_indices]
                top3_scores = [avg_scores[i].item() for i in top3_indices]
                print(f"  Top 3: {top3_letters[0]}({top3_scores[0]:.3f}), {top3_letters[1]}({top3_scores[1]:.3f}), {top3_letters[2]}({top3_scores[2]:.3f})")

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
    out_csv = f"model_choices-clip-improved-{opt.experiment_name}-{opt.history_type}.csv"
    
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