import argparse
import json
import random
from pathlib import Path

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from torch.utils.data import DataLoader


def load_triplets(path: Path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # InputExample with 3 texts: MultipleNegativesRankingLoss treats
            # texts[0]=anchor, texts[1]=positive, texts[2:]=extra hard negatives
            examples.append(InputExample(texts=[row["query"], row["positive"], row["negative"]]))
    return examples


def split_train_eval(examples, eval_fraction: float = 0.02, seed: int = 42):
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_eval = max(50, int(len(shuffled) * eval_fraction))
    return shuffled[n_eval:], shuffled[:n_eval]


def build_triplet_evaluator(eval_examples):
    # TripletEvaluator checks: is cosine_sim(anchor, positive) > cosine_sim(anchor, negative)?
    anchors = [e.texts[0] for e in eval_examples]
    positives = [e.texts[1] for e in eval_examples]
    negatives = [e.texts[2] for e in eval_examples]
    return evaluation.TripletEvaluator(anchors, positives, negatives, name="msmarco_xi_eval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/finetune/msmarco_xi_triplets.jsonl")
    parser.add_argument("--base-model", type=str,
                         default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                         help="Small multilingual model, fine on CPU/laptop/free Colab GPU. "
                              "Swap to 'BAAI/bge-m3' if you have a real GPU and want max quality.")
    parser.add_argument("--output", type=str, default="models/embedder-indic-finetuned")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    args = parser.parse_args()

    data_path = Path(args.data)
    print(f"Loading triplets from {data_path} ...")
    examples = load_triplets(data_path)
    print(f"Loaded {len(examples)} triplets")

    train_examples, eval_examples = split_train_eval(examples, args.eval_fraction)
    print(f"Train: {len(train_examples)}  |  Eval: {len(eval_examples)}")

    print(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    evaluator = build_triplet_evaluator(eval_examples)

    def extract_score(result):
        # Newer sentence-transformers versions return a dict of metrics instead
        # of a single float; evaluator.primary_metric tells us which key to read.
        if isinstance(result, dict):
            key = getattr(evaluator, "primary_metric", None)
            if key and key in result:
                return result[key]
            return next(iter(result.values()))  # fall back to first metric
        return result

    print("\n--- Baseline eval (before fine-tuning) ---")
    baseline_score = extract_score(evaluator(model))
    print(f"Baseline triplet accuracy: {baseline_score:.4f}")

    warmup_steps = int(len(train_loader) * args.epochs * 0.1)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output,
        show_progress_bar=True,
    )

    print("\n--- Final eval (after fine-tuning) ---")
    final_score = extract_score(evaluator(model))
    print(f"Final triplet accuracy: {final_score:.4f}  (baseline was {baseline_score:.4f})")
    print(f"\nModel saved to: {args.output}")
    print("Next: point src/embedding/encode.py at this checkpoint to embed your document chunks.")


if __name__ == "__main__":
    main()