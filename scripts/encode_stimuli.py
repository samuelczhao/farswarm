"""Encode earnings call transcripts with real TRIBE v2 brain model.

Run with the tribe venv:
    .venv-tribe/bin/python scripts/encode_stimuli.py

Produces .npy files in benchmarks/data/neural_responses/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

STIMULI_DIR = Path("benchmarks/data/stimuli")
OUTPUT_DIR = Path("benchmarks/data/neural_responses")


def main() -> None:
    if "HF_TOKEN" not in os.environ:
        print("ERROR: Set HF_TOKEN environment variable")
        print("  export HF_TOKEN=hf_your_token_here")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading TRIBE v2 model...")
    from tribev2 import TribeModel

    cache = Path("./cache/tribe_v2")
    cache.mkdir(parents=True, exist_ok=True)
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache)
    print("Model loaded.\n")

    stimuli = sorted(STIMULI_DIR.glob("*.txt"))
    print(f"Found {len(stimuli)} stimulus files\n")

    for stim_path in stimuli:
        out_path = OUTPUT_DIR / f"{stim_path.stem}_neural.npy"
        if out_path.exists():
            print(f"  SKIP {stim_path.name} (already encoded)")
            continue

        print(f"  Encoding: {stim_path.name}")
        try:
            df = model.get_events_dataframe(text_path=str(stim_path))
            preds, segments = model.predict(events=df)
            print(f"    Shape: {preds.shape}")
            print(f"    Mean: {np.mean(preds):.4f}, Max: {np.max(preds):.4f}")

            np.save(out_path, preds.astype(np.float32))
            print(f"    Saved: {out_path}")
        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()

    print(f"\nDone. Files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
