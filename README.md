# cosmos-exploration

## Build
```
cd cosmos-reason2 && docker build -t cosmos-reason2 -f ./Dockerfile .  && cd ..
cd synthetic-vqa && docker build --build-arg CUDA_VERSION=12.8.0 -t synthetic-vqa . && cd ..

docker tag cosmos-reason2:latest embodied-ai-gitea.apps.octocp.alexnet.htx/runai-embodied-ai/cosmos-reason2:latest
docker tag synthetic-vqa:latest embodied-ai-gitea.apps.octocp.alexnet.htx/runai-embodied-ai/synthetic-vqa:latest

docker images --format '{{.Repository}}:{{.Tag}}' | grep '^embodied-ai-gitea' | while read img; do docker push "$img"; done
```

## Deploy
```
oc apply -f synthetic-vqa/configmap-prompts.yaml
oc apply -f synthetic-vqa/pod-synthetic-vqa.yaml
```