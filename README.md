to start docker image

sudo docker run --gpus all --rm -it \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v ~/model_repository:/home/upanishad/SDInferenceServer/model_repository \
  nvcr.io/nvidia/tritonserver:23.09-py3 \
  tritonserver --model-repository=/home/upanishad/SDInferenceServer/model_repository --log-verbose=1
