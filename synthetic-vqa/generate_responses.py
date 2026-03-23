"""
Generate VLM reasoning responses for synthetic images.

Reads the manifest.json produced by generate_images.py, sends each image + questions
to a Cosmos-Reason2 VLM via its OpenAI-compatible API, and outputs a responses.json file.

Usage:
    python generate_responses.py --manifest-file ./output/manifest.json --output-dir ./output --base-url http://localhost:8000/v1 --api-key token-abc123
    python generate_responses.py --manifest-file ./output/manifest.json --output-dir ./output --base-url http://localhost:8000/v1 --api-key token-abc123 --model-id nvidia/Cosmos-Reason2-8B

Dependencies:
    pip install openai
"""

import argparse
import base64
import json
import logging
import time
from pathlib import Path

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "nvidia/Cosmos-Reason2-2B"

SYSTEM_PROMPT = """\
You are a visual reasoning assistant. When given an image and a question, \
think step by step about what you observe, then provide a concise answer in English.

You MUST format your response exactly as:
<think>your reasoning here</think>
<answer>your answer here</answer>"""

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_vlm(client: OpenAI, model: str, image_path: str, question: str) -> str:
    image_b64 = encode_image_base64(image_path)
    ext = Path(image_path).suffix.lstrip(".").lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                    {"type": "text", "text": question},
                ],
            },
        ],
    )
    # logger.info(f"Raw response: {response.choices[0].message}") 
    return response.choices[0].message.reasoning, response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Generate VLM responses for synthetic images")
    parser.add_argument("--manifest-file", required=True, help="Path to manifest.json from generate_images.py")
    parser.add_argument("--output-dir", required=True, help="Path to output directory for responses.json")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="VLM model ID")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--api-key", default="token-unused", help="API key for the endpoint (default: token-unused)")
    parser.add_argument("--questions-file", required=True, help="JSON file with questions (list of strings), e.g. example_questions.json")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # Verify the endpoint is reachable
    try:
        client.models.list()
        logger.info(f"Endpoint {args.base_url} is up")
    except Exception as e:
        logger.error(f"Cannot reach endpoint {args.base_url}: {e}")
        raise SystemExit(1)

    with open(args.manifest_file) as f:
        manifest = json.load(f)
    logger.info(f"Loaded {len(manifest)} entries from {args.manifest_file}")

    with open(args.questions_file) as f:
        questions = json.load(f)
    logger.info(f"Loaded {len(questions)} questions from {args.questions_file}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load existing progress if exists (resumable)
    rows = []
    seen = set()
    responses_file = output_path / "responses.json"
    if responses_file.exists():
        with open(responses_file) as f:
            rows = json.load(f)
        seen = {r["id"] for r in rows}
        logger.info(f"Resuming: {len(rows)} entries already generated")

    total = len(manifest)
    done = len(seen)

    for entry in manifest:
        image_path = entry["image"]
        prompt_id = entry["id"]

        if prompt_id in seen:
            continue

        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}, skipping {prompt_id}")
            continue

        done += 1
        logger.info(f"[{done}/{total}] {prompt_id}")

        trace_parts = []
        for i, question in enumerate(questions):
            logger.info(f"  Question {i+1}/{len(questions)}: {question[:60]}...")

            while True:
                try:
                    reasoning, response = query_vlm(client, args.model_id, image_path, question)
                    trace_parts.append(reasoning if reasoning else "")
                    # trace_parts.append(response if response else "")
                    break
                except Exception as e:
                    logger.warning(f"Failed for {prompt_id} q{i+1}, retrying in 5s: {e}")
                    time.sleep(5)

        reasoning = "\n\n".join(trace_parts)

        rows.append({
            "id": prompt_id,
            "image": image_path,
            "reasoning": reasoning,
        })

        # Save after each manifest entry (resumable)
        with open(responses_file, "w") as f:
            json.dump(rows, f, indent=2)

    logger.info(f"Done. {len(rows)} entries saved to {responses_file}")


if __name__ == "__main__":
    main()
