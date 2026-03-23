# cosmos-exploration

## Editor
```
# Install vscode, with `ms-azuretools.vscode-docker` and `ms-vscode-remote.remote-containers`
apt install jq -y
```

## Build
```
cd cosmos-reason2 && docker build -t cosmos-reason2 -f ./Dockerfile . && cd ..
cd synthetic-vqa && docker build --build-arg CUDA_VERSION=12.8.0 -t synthetic-vqa . && cd ..

docker tag cosmos-reason2:latest embodied-ai-gitea.apps.octocp.alexnet.htx/runai-embodied-ai/cosmos-reason2:latest
docker tag synthetic-vqa:latest embodied-ai-gitea.apps.octocp.alexnet.htx/runai-embodied-ai/synthetic-vqa:latest

docker images --format '{{.Repository}}:{{.Tag}}' | grep '^embodied-ai-gitea' | while read img; do docker push "$img"; done
docker volume create synthetic-vqa-output
```

## Deploy (Local)
```
# Launch Reasoning Model Server (or use a public model)
uv run --with vllm vllm serve nvidia/Cosmos-Reason2-2B \
  --allowed-local-media-path "$(pwd)" \
  --max-model-len 8192 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 \
  --port 8000

# Test the server with an image and prompt
bash test_endpoint.sh ./hazmat_fire.png "you are a robot tasked to collect samples for analysis. Think step by step about what you observe, provide a concise plan as a numbered list of actions to collect a sample of material in this scene for analysis. You MUST format your response exactly as: <think>your reasoning here</think><answer>your answer here</answer>. You MUST format your answer as exactly ONE of the following: 1. move to (object) OR 2. pick up (object)"  | jq 
```

## Dataset Generation
```
# Generate Images. Inspect using vscode Docker Extension using Dev Containers
docker run --gpus all \
  -v synthetic-vqa-output:/workspace/output \
  -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
  synthetic-vqa generate_images.py --prompts-file example_prompts.json --output-dir /workspace/output

# Generate Annotations
docker run --gpus all \
  --network host \
  -v synthetic-vqa-output:/workspace/output \
  synthetic-vqa generate_responses.py \
    --manifest-file /workspace/output/manifest.json \
    --questions-file example_questions.json \
    --actions-file example_actions.json \
    --output-dir /workspace/output \
    --base-url http://localhost:8000/v1
```

## Edit Dataset