#!/bin/bash
set -e

if ! command -v jq &> /dev/null; then
  echo "jq not found, installing..."
  sudo apt-get update -qq && sudo apt-get install -y -qq jq
fi

if [ $# -lt 2 ]; then
  echo "Usage: $0 <image_path> <prompt>"
  exit 1
fi

export IMAGE_PATH="$1"
export PROMPT="$2"
export BASE_URL="${BASE_URL:-http://localhost:8000}"
export MODEL="${MODEL:-nvidia/Cosmos-Reason2-2B}"

python3 -c "
import base64, json, os, sys
with open(os.environ['IMAGE_PATH'], 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()
payload = {
    'model': os.environ['MODEL'],
    'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
        {'type': 'text', 'text': os.environ['PROMPT']}
    ]}]
}
json.dump(payload, sys.stdout)
" | curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @- | jq .
