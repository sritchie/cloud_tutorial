# Cloud Tutorial (aka "Cloudssifier")

Simple classifier trained on Cloud! This repository contains a workflow that
trains a vanilla fully-connected neural network to predict MNIST digits, using
Tensorflow 2.0's SGD optimizer.

## Run locally

Run the following from the `cloud_tutorial` project directory:

```bash
pip install . # Install all dependencies locally
python cloudssifier/train.py
```

If you pass in `--help`, you'll see a number of options that you can pass to
customize the behavior of the MNIST classifier.

```bash
$ python cloudssifier/train.py --help

Train a classifier on MNIST
flags:

cloudssifier/train.py:
  --batch_size: training batch size.
    (default: '64')
    (an integer)
  --data_path: path to mnist npz file.
    (default: 'data/mnist.npz')
  --epochs: number of training epochs.
    (default: '10')
    (an integer)
  --job-dir: Catch job directory passed by cloud.
    (default: '')

Try --helpfull to get a list of all flags.
```

## Setup for Cloud

First, follow the instructions
[here](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-keras)
to get set up with:

-   a Cloud project,
-   your very own Cloud bucket,
-   the Google Cloud SDK, and
-   the required credentials to interact with Cloud from your local machine.

By the time you're through you should have the following environment variables
set:

```bash
export PROJECT_ID=<your cloud project>
export BUCKET_NAME=<name of your bucket>
export REGION="us-central1" # Unless you have some great reason to choose another
```

### Move MNIST data to your cloud bucket

From the `cloud_tutorial` directory, run the following command:

```bash
DATA_PATH="gs://$BUCKET_NAME/data/mnist.npz" bash -c 'gsutil cp data/mnist.npz $DATA_PATH'
```

For the GPU jobs you will need to make sure you have quota in `$REGION`. The
examples below execute with P100 GPUs.

## Submit via command line

This example submits the model training job to AI Platform, where it will
execute using CPUs. More more info on the command's arguments, see this
[AI platform page](https://cloud.google.com/sdk/gcloud/reference/ai-platform/jobs/submit/training).

From the `cloud_tutorial` directory, run:

```bash
JOB_NAME="MNIST_training_cpu_${USER}_$(date +%Y%m%d_%H%M%S)"
JOB_DIR="gs://$BUCKET_NAME/mnist_demo"

gcloud ai-platform jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --staging-bucket gs://$BUCKET_NAME \
  --module-name cloudssifier.train \
  --package-path cloudssifier/ \
  --region $REGION \
  -- \
  --data_path=$DATA_PATH
```

You'll be able to see your job running at the
https://console.cloud.google.com/ai-platform/jobs dashboard. You can also stream
the logs directly to the terminal by running:

```bash
gcloud ai-platform jobs stream-logs $JOB_NAME
```

## Submit a GPU job

To submit a GPU job:

1.  delete tensorflow from the `REQUIRED_PACKAGES` line in `setup.py`
2.  Run the following command from the `cloud_tutorial` root:

```bash
JOB_NAME="MNIST_training_GPU_${USER}_$(date +%Y%m%d_%H%M%S)"
JOB_DIR="gs://$BUCKET_NAME/mnist_demo"

gcloud ai-platform jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --staging-bucket gs://$BUCKET_NAME \
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

The file `submit.py` contains an example GPU submission using the Python api.
Run the following, again from `cloud_tutorial`:

```bash
python submit.py \
  --bucket gs://$BUCKET_NAME \
  --project_id $PROJECT_ID \
  --region $REGION
```
