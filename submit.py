"""Submit a cloud ml job using python!"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import datetime

from absl import app
from absl import flags
from absl import logging

from googleapiclient import discovery
from googleapiclient import errors
from google.cloud import storage
from setuptools.sandbox import run_setup

# Structure from:
#  https://cloud.google.com/ml-engine/docs/tensorflow/python-client-library
#  https://cloud.google.com/ml-engine/docs/tensorflow/training-jobs

FLAGS = flags.FLAGS

# Configuration:
flags.DEFINE_string('job_name', 'MNIST_Classifier',
                    'A simple fully connected MNIST classifier.')
flags.DEFINE_string('region', 'us-west1', 'GCE region.')
flags.DEFINE_string('project_id', None, 'GCE project id.')
flags.DEFINE_string('bucket', None, 'Path to staging bucket.')


def package_and_upload():
  """Package training application and upload to GCE bucket.

  Outputs:
    package_uris - URI of package in GCE bucket.
  """
  logging.info('Packaging and uploading.')
  BUCKET = FLAGS.bucket.split('/')[-1]

  # Package
  run_setup('setup.py', ['sdist'])

  # Upload
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(BUCKET)
  dist_dir = 'dist'
  if not os.path.exists(dist_dir):
    raise IOError('Missing dist directory (was supposed to be generated)')
  package_uris = []
  for src in glob.glob(dist_dir + '/*'):
    dest = os.path.join('staging', os.path.relpath(src, dist_dir))
    blob = bucket.blob(dest)
    blob.upload_from_filename(src)
    package_uris.append(os.path.join('gs://' + BUCKET, dest))

  logging.info('Package URIs: %s' % ', '.join(package_uris))

  return package_uris


def submit_job(package_uris):
  """Submit job to GCE.

  Inputs:
    package_uris - URI of package in GCE bucket.
  """
  logging.info('Submitting job.')

  time_stamp = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
  JOB_NAME = FLAGS.job_name + time_stamp

  USR_PROJECT_ID = FLAGS.project_id
  BUCKET = FLAGS.bucket.split('/')[-1]
  JOB_DIR = os.path.join('gs://', BUCKET, 'mnist_demo')
  REGION = FLAGS.region
  DATA_PATH = os.path.join('gs://', BUCKET, 'data/mnist.npz')
  PACKAGE_URIS = package_uris

  training_inputs = {
      'packageUris': PACKAGE_URIS,
      'pythonModule': 'cloudssifier.train',
      'args': ['--data_path', DATA_PATH],
      'region': REGION,
      'jobDir': JOB_DIR,
      'scaleTier': 'CUSTOM',
      'masterType': 'standard_p100',
      'runtimeVersion': '1.12',
      'pythonVersion': '3.5',
  }

  job_spec = {'jobId': JOB_NAME, 'trainingInput': training_inputs}

  # Store your full project ID in a variable in the format the API needs.
  project_id = 'projects/{}'.format(USR_PROJECT_ID)

  # Build a representation of the Cloud ML API.
  ml = discovery.build('ml', 'v1')

  # Create a request to call projects.models.create.
  request = ml.projects().jobs().create(body=job_spec, parent=project_id)

  # Make the call.
  try:
    response = request.execute()
    logging.info(response)
    print('\nTo stream logs run: gcloud ai-platform jobs stream-logs %s\n' %
          JOB_NAME)
  except errors.HttpError as err:
    # Something went wrong, print out some information.
    logging.error('There was an error creating the model. Check the details:')
    logging.error(err._get_reason())


def main(argv):
  logging.info('Running main.')

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  package_uris = package_and_upload()
  submit_job(package_uris)


if __name__ == '__main__':
  flags.mark_flag_as_required('project_id')
  flags.mark_flag_as_required('bucket')
  app.run(main)
