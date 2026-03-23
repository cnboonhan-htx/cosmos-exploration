"""
Create a LLaVA-style HuggingFace dataset from responses.json.

Reads the responses.json produced by generate_responses.py, pairs each image with
its reasoning and action plan, and builds a HuggingFace dataset matching the
trl-lib/llava-instruct-mix schema (same as cnboonhan-htx/reasoning-cookbook).

Usage:
    python create_llava_dataset.py --responses-file ./output/responses.json --output-dir ./output --dataset-name cnboonhan-htx/demo_hazmat

Dependencies:
    pip install datasets Pillow
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_FEATURES = Features({
    "images": [Image(decode=True)],
    "prompt": [{"role": Value("string"), "content": Value("string")}],
    "completion": [{"role": Value("string"), "content": Value("string")}],
})


def main():
    parser = argparse.ArgumentParser(description="Create a LLaVA-style dataset from responses.json")
    parser.add_argument("--responses-file", required=True, help="Path to responses.json from generate_responses.py")
    parser.add_argument("--output-dir", required=True, help="Path to output HuggingFace dataset directory")
    parser.add_argument("--dataset-name", required=True, help="Name for the dataset")
    args = parser.parse_args()

    with open(args.responses_file) as f:
        responses = json.load(f)
    logger.info(f"Loaded {len(responses)} entries from {args.responses_file}")

    output_path = Path(args.output_dir) / args.dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    image_lists = []
    prompts = []
    completions = []

    for entry in responses:
        image_path = Path(entry["image"])
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}, skipping {entry['id']}")
            continue

        # Load image as PIL so it gets embedded as bytes in the dataset
        img = PILImage.open(image_path).convert("RGB")
        image_lists.append([img])

        # Build the user prompt from reasoning context
        user_content = (
            "Based on the visual reasoning about this scene, "
            "what sequence of actions should be taken?\n\n"
            f"Reasoning:\n{entry['reasoning']}"
        )
        prompts.append([{"role": "user", "content": user_content}])
        completions.append([{"role": "assistant", "content": entry.get("actions", "")}])

    logger.info(f"Building dataset with {len(image_lists)} entries...")

    dataset_dict = {
        "images": image_lists,
        "prompt": prompts,
        "completion": completions,
    }
    ds = Dataset.from_dict(dataset_dict, features=DATASET_FEATURES)

    # Save as Parquet in HuggingFace-expected directory structure
    train_dir = output_path / "default" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = train_dir / "0000.parquet"
    ds.to_parquet(str(parquet_path))

    logger.info(f"Dataset '{args.dataset_name}' saved to {parquet_path} ({len(ds)} rows)")
    logger.info(f"Schema: {ds.features}")


if __name__ == "__main__":
    main()
