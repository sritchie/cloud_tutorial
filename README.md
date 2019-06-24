# cloud_tutorial
Simple classifier trained on cloud!

## Run locally
From cloud_tutorial:
```bash
cd cloudssifier/cloudssifier
python train.py
```

## Setup for cloud

Hopefully you have already setup a project (`$PROJECT_ID`), storage bucket ( `$STAGING_BUCKET`), and installed the command line tools and python SDK. If not, follow instructions [here](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-keras).

* Move MNIST data into staging bucket

From cloud_tutorial:
```bash
DATA_PATH="$STAGING_BUCKET/data/mnist.npz"
gsutil cp data/mnist.npz $DATA_PATH
```
* For the gpu jobs you will need to make sure you have quota in `$REGION`. The examples below are with P100 gpus.


## Submit via command line
From cloud_tutorial
```bash
cd cloudssifier
JOB_NAME='MNIST_training_cpu'
JOB_DIR="$STAGING_BUCKET/mnist_demo"

python setup.py sdist
gcloud ai-platform jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --staging-bucket $STAGING_BUCKET \
  --module-name cloudssifier.train \
  --package-path cloudssifier/ \
  --region $REGION\
  -- \
  --data_path=$DATA_PATH

gcloud ai-platform jobs stream-logs $JOB_NAME
```

To submit a gpu job run:

```bash
JOB_NAME='MNIST_training_gpu'

python setup.py sdist
gcloud beta ai-platform jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --staging-bucket $STAGING_BUCKET \
  --module-name cloudssifier.train \
  --package-path cloudssifier/ \
  --region $REGION\
  --scale-tier custom \
  --master-machine-type standard_p100 \
  --runtime-version 1.12 \
  --python-version 3.5 \
  -- \
  --data_path=$DATA_PATH

gcloud ai-platform jobs stream-logs $JOB_NAME

```

## Submit via python api
The file submit.py contains an example gpu submission using the python api.
```bash
python submit.py \
  --bucket $STAGING_BUCKET \
  --project_id $PROJECT_ID \
  --region $REGION
```
