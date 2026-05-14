import argparse
import json
from pathlib import Path

import torch

import backend.main as main
from backend.evaluate import evaluate_dataset


def evaluate_checkpoint(
    checkpoint: Path,
    masked_root: Path,
    original_root: Path,
    limit: int | None,
    output: Path,
) -> dict:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    main.degan_model = main.DEGAN().to(main.device)
    loaded = main._load_model_weights(main.degan_model, checkpoint.name)
    if not loaded:
        state_dict = torch.load(checkpoint, map_location=main.device)
        main.degan_model.load_state_dict(state_dict, strict=False)
    main.degan_model.eval()

    results = evaluate_dataset(masked_root, original_root, limit)
    results["checkpoint"] = str(checkpoint.resolve())

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a specific DEGAN checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a DEGAN checkpoint")
    parser.add_argument("--masked", type=str, required=True, help="Masked dataset root")
    parser.add_argument("--original", type=str, required=True, help="Original dataset root")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--output", type=str, required=True, help="Metrics JSON output path")
    args = parser.parse_args()

    results = evaluate_checkpoint(
        Path(args.checkpoint),
        Path(args.masked),
        Path(args.original),
        args.limit,
        Path(args.output),
    )
    for key, value in results.items():
        print(f"{key}: {value}")
