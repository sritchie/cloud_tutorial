#!/bin/bash

JOB_NAME="cloud_hello_world_`date +"%b%d_%H%M%S"`"
TAG="gcr.io/research-3141/cloud-hello-world"
REGION="us-central1"
DOCKERFILE=Dockerfile

# --- Chose the machine you want to run on (CPU vs. GPU, ~regions)
# https://cloud.google.com/ml-engine/docs/tensorflow/machine-types
# https://cloud.google.com/ml-engine/docs/tensorflow/regions
#
export MACHINE_TYPE="--scale-tier CUSTOM --master-machine-type standard_p100"
#export MACHINE_TYPE="--scale-tier basic_gpu"
#export MACHINE_TYPE="--scale-tier basic"

# --- Launch different types of training
MODE=${1:-remote}
case "$MODE" in
  help)
    echo "Usage:"
    echo "  ./submit.sh {build, shell, local, remote, remote-nodocker, gke} \n"
    ;;

  local)
    echo "Building and launching locally"

    echo ">>> nvidia-docker build"
    nvidia-docker build -f $DOCKERFILE -t $TAG $PWD

    # NOTE: can also use "nvidia-docker run"
    echo ">>> docker run"
    docker run --runtime=nvidia $TAG
    ;;

  remote)
    echo "Launching remotely on ai-platform"
    echo "Job name: $JOB_NAME"

    #--- Build
    echo ">>> nvidia-docker build"
    nvidia-docker build -f $DOCKERFILE -t $TAG $PWD

    #--- Authenticate (just to be safe)
    echo ">>> gcloud auth configure-docker"
    gcloud auth configure-docker

    #--- Upload
    echo ">>> docker push"
    docker push $TAG

    #--- Launch
    echo ">>> gcloud beta ai-platform jobs submit"
    gcloud beta ai-platform jobs submit training $JOB_NAME \
      --region $REGION \
      --master-image-uri $TAG \
      $MACHINE_TYPE

    # --- Turn on streaming automatically
    echo ">>> gcloud ai-platform jobs stream-logs"
    gcloud ai-platform jobs stream-logs $JOB_NAME
    ;;

  build)
    echo "Building"
    nvidia-docker build -f $DOCKERFILE -t $TAG $PWD
    ;;

  shell)
    echo "Building and opening local shell"
    nvidia-docker build -f $DOCKERFILE -t $TAG $PWD
    docker run --entrypoint /bin/bash --runtime=nvidia -it $TAG
    ;;

  remote-nodocker)
    echo "Launching remotely on ai-platform (without installing docker) (SLOW!)"
    echo "Job name: $JOB_NAME"

    #--- Build and upload
    gcloud builds submit -t $TAG .

    gcloud beta ai-platform jobs submit training $JOB_NAME \
      --region $REGION \
      --master-image-uri $TAG \
      $MACHINE_TYPE

    gcloud ai-platform jobs stream-logs $JOB_NAME
    ;;

  gke)
    echo "Launching remotely on GKE"

    echo ">>> nvidia-docker build"
    nvidia-docker build -f $DOCKERFILE -t $TAG $PWD

    echo ">>> gcloud auth configure-docker"
    gcloud auth configure-docker

    echo ">>> docker push"
    docker push $TAG

    kubectl apply -f job-config.yaml
    ;;
esac
