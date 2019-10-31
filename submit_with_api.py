"""Submit a cloud ml job using python"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import datetime

from absl import app
from absl import flags
from absl import logging

from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors
from google.cloud import storage
import itertools
from setuptools.sandbox import run_setup
import logging
import numpy as np
import yaml, subprocess
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

#### USAGE 
### GENERAL AI PLATFORM JOB SUBMISSION SCRIPT FOR DOCKER BUT ALSO USUAL PACKAGE UPLOADING.
### Examples of use: 
##  Submitting directly to cloud through a package (remote or locally):
##  python3 submit_with_api.py --config_files=experiment_config.yaml --package remote ##
##  Trying submitting docker configuration with --docker local, remote 
##  python3 submit_with_api.py --config_files=experiment_config.yaml --docker local ## 


#### FLAGS.config_file for the experiment specifics. In particular uses_metaconfig is the direction of the general experiment setup.
#### Can send arguments to the jobs by adding them to the config.yaml or directly via FLAGS.args=arg1,arg2,arg3
#### config.yaml: args to send, uses_metaconfig for metaconfig, values of pid (see below) and desired number of jobs (numjobs).
#### metaconfig contains the cloud and docker/package configuration and the parameter one is looping around .

flags.DEFINE_string('config_file',None,'Configuration for particular experiment')
flags.DEFINE_string('args',None,'If one wants to add more arguments to the python script to send. Write them as arg1,arg2,arg3,... with arg1: -sthg1=value1, ...')
### THESE ARE THE METACONFIG ARGS
METACONFIG_ARGS=['uses_metaconfig','job_name','numjobs','docker_tag','DOCKERFILE','hypertype','DOCKERFILEGPU',
'setup_loc','module','runtime_version','pythonVersion','region','bucket_id','project_id','pid']
#### Can submit multiple jobs by giving it a parameter 'pid'. The script will call jobs with various pids. 
#### The pids are sent in batches, so the job should admit a list of pids to be received in parallel.
### Example:
##  metaconfig  
##  > pid: lr
##  config
##  > lrlist: [0.1,0.2,0.4,0.7,0.8]
##  > numjobs: 3
##  This will sent 3 jobs with -lr_list=0.1,0.2 ; -lr-list=0.4,0.7 ; ... 


## DOCKER SPECIFIC
## To use with docker one has to specify DOCKERFILE, DOCKERFILEGPU,DOCKER_TAG in the metaconfig
## keep in mind that to save things in a bucket one has to transfer the credentials somehow to the docker. 
## --docker accepts local, local-cache or remote configurations
flags.DEFINE_string('docker',None,'Mode of execution for docker local, remote, ...')
flags.DEFINE_boolean('build',True,'Avoid building and submit the cached/uploaded image.')
## PACKAGE SPECIFIC
flags.DEFINE_string('package',None,'Mode of execution for package local, remote, ...')





## Sending GPU, be mindful about sending gpu docker jobs via mac since one can't build it. 
flags.DEFINE_boolean('GPU',True,'Is it going to use remotely GPU? Keep in mind the dockerfile!')
FLAGS = flags.FLAGS



def package_and_upload(metaconfig):
  """Package training application and upload to GCE bucket.

  Outputs:
    package_uris - URI of package in GCE bucket.
  """
  logging.info('Packaging and uploading.')
  
  if 'setup_loc' in metaconfig:
    setup_dir=os.getcwd()+'/'+metaconfig['setup_loc']+'/'
  else:
    setup_dir=''

  # Package
  run_setup(setup_dir+'setup.py', ['sdist'])
  # Upload
  storage_client = storage.Client(project=metaconfig['project_id'])
  bucket = storage_client.get_bucket(metaconfig['bucket_id'])
  print(metaconfig)
  dist_dir=setup_dir+'dist'

  if not os.path.exists(dist_dir):
    raise IOError('Missing dist directory (was supposed to be generated)')
  package_uris = []
  for src in glob.glob(dist_dir + '/*'):
    dest = os.path.join('staging', os.path.relpath(src, dist_dir))
    blob = bucket.blob(dest)
    blob.upload_from_filename(src)
    package_uris.append(os.path.join('gs://' + metaconfig['bucket_id'], dest))

  logging.info('Package URIs: %s' % ', '.join(package_uris))

  return package_uris



def submit_job(config,args,package_uris=None):
  """Submit job to GCE.

  Inputs:
    package_uris - URI of package in GCE bucket.
  """
  logging.info('Submitting job.')
  
  if FLAGS.docker is not None:
      training_inputs={'masterConfig':{'imageUri':config['docker_tag']}}
  else:
      try:
        training_inputs = {'packageUris': package_uris, 'pythonModule': config['module'],'runtimeVersion': config['runtime_version'],'pythonVersion': config['pythonVersion']}
      except:
        raise app.UsageError('This is not a docker task and the packages are not configured properly')



  training_inputs['args']= args
  training_inputs['region'] = config['region']
  training_inputs['jobDir']='gs://'+config['bucket_id']

  if not FLAGS.GPU:
    training_inputs["scaleTier"]= "CUSTOM"
    training_inputs['masterType']= 'complex_model_l'  
  elif FLAGS.GPU:
    training_inputs["scaleTier"]= "CUSTOM"
    training_inputs['masterType']= 'standard_p100'


  job_spec = {'jobId': config['job_name'] + datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S'), 'trainingInput': training_inputs}
  # Store your full project ID in a variable in the format the API needs.
  project_id = 'projects/{}'.format(config['project_id'])
  # Build a representation of the Cloud ML API.
  ml = discovery.build('ml', 'v1')
  # Create a request to call projects.models.create.
  
  request = ml.projects().jobs().create(body=job_spec, parent=project_id)

  # Make the call.
  try:
    response = request.execute()
    logging.info(response)
  except errors.HttpError as err:
    # Something went wrong, print out some information.
    logging.error('There was an error creating the model. Check the details:')
    logging.error(err._get_reason())

def build_docker(config,docker,GPU):
## THIS COULD BE CLEANED.

  if docker=='local-nocache':
    subprocess.call(['docker', 'build','--no-cache','-f', config['DOCKERFILE'],'-t',config['docker_tag'],os.getcwd()])

  if GPU:
    # try:
    print('Using Nvidia docker')
    ### A LITTEL CONFUSING THIS
#    pipe=subprocess.Popen(['nvidia-docker', 'build','--no-cache','-f', config['DOCKERFILEGPU'],'-t',config['docker_tag'],os.getcwd()],shell=True, stdout = subprocess.PIPE) 
    print(' '.join(['nvidia-docker', 'build','--no-cache','-f', config['DOCKERFILEGPU'],'-t',config['docker_tag'],os.getcwd()]))
    try:
      subprocess.call(['nvidia-docker', 'build','--no-cache','-f', config['DOCKERFILEGPU'],'-t',config['docker_tag'],os.getcwd()]) 
    except:
    
      raise ValueError('This requires nvidia-docker, install if using linux or submit the loaded image using --nobuild.')
      
  else:

    subprocess.call(['docker', 'build','-f', config['DOCKERFILE'],'-t',config['docker_tag'],os.getcwd()])
  
  if docker=='remote':  
    subprocess.call(['docker','push',config['docker_tag']])

def main(argv):
    logging.info('Running main.')
    if not FLAGS.config_file:
      raise app.UsageError('Needs a proper configfile')

    with open(FLAGS.config_file) as f:   
        config = yaml.load(f, Loader=yaml.FullLoader)
 
    ### load respective metaconfig unless file has all details. 
    if 'uses_metaconfig' in config:
      with open(config['uses_metaconfig']) as f:      
          metaconfig = yaml.load(f, Loader=yaml.FullLoader)
    else:
      metaconfig= config

    package_uris=None

    if FLAGS.docker is not None and FLAGS.build:
      build_docker(metaconfig,FLAGS.docker,FLAGS.GPU)
    elif FLAGS.package=='remote':
        package_uris=package_and_upload(metaconfig)

    ### LOOP OVER A PARAMETER pid AND DISTRIBUTE ITS VALUES AMONG JOBS
    ### TODO CONSIDER 'pid' A LIST OF PARAMETERS
    not_args=METACONFIG_ARGS
    args0=[]
    other_args=[]
    ### HAVE TO DEBUG THIS.
    if 'pid' in metaconfig:
      pids=metaconfig['pid']
      par_list=[]
      if isinstance(pids,str):
        name=pids
        pids=[pids]
      elif len(pids)==1:
        name=pids[0]
      else:
        name='loopvar'
        #args0+=['-pids=',','.join(pids)]
      

      for pid in pids:
        not_args+=['min'+pid,'max'+pid,'num'+pid,pid+'_list','hyper'+pid]
        hyper=None
        if 'hyper'+pid in metaconfig:
          hyper=metaconfig['hyper'+pid]
        elif 'hyper'+pid in config:
          hyper=config['hyper'+pid]
        else:
          print('Should specify hyper'+pid)
          continue
        
        if hyper=='list':
          par_list.append( config[pid+'_list'] )
        else:

          if config['max'+pid]==-1:
            if config['num_samples']==-1:
              raise ValueError('Not supported.')
            else:
              print('Setting bs to number of samples')
              config['max'+pid]=config['num_samples']
            
            
          if hyper=='log2':
            par_list.append(np.logspace(np.log2(config['min'+pid]), np.log2(config['max'+pid]), num=config['num'+pid],base=2) )
          elif hyper=='lin':
            par_list.append( np.linspace(config['min'+pid], config['max'+pid], num=config['num'+pid]) )
        

      par_list=[list(el) for el in list(itertools.product(*par_list))]

      n = (len(par_list)-1)//config['numjobs'] + 1 
      par_listf = [par_list[i*n:(i+1) * n] for i in range((len(par_list) + n - 1) // n )]
      ## THIS IS AN UGLY WAY OF SENDING THIS AS STRING. WHEN THERE IS ONLY ONE IT IS JUST SEPARATED BY COMMAS

      par_listf= [['+'.join([str(elc) for elc in elb]) for elb in el ]for el in par_listf ] 
      print('Total of ',len(par_list),' jobs')
    else:
      par_listf=[0]
    ### PREPARATION OF ARGUMENTS TO SEND

    if FLAGS.args:
      args0+= FLAGS.args.split(',')
      
    args0+=['-'+el+'='+str(config[el]) for el in config if el not in not_args]
    args0=['-other_args='+'??'.join(args0)]
    k=0

    for par in par_listf:
        k+=1
        
        if 'pid' in metaconfig:
          pids=metaconfig['pid']
          if isinstance(pids,str):
            name=pids
          elif len(pids)==1:
            name=pids[0]
          else:
            name='loopvar'
          

          print("")
          print('>> Sending job {:.0f} out of {:.0f}'.format(k,len(par_listf))+' with '+name+':',par) 
          args=other_args+args0+['-'+name+'_list='+','.join(par)]
          print("")
        else:
          args=other_args+args0
        if k==len(par_listf) and 'special_last' in config:
          args+=[config['special_last']]
      
        if FLAGS.docker=='remote' or FLAGS.package=='remote':
            if 'job_name' in config:
              metaconfig['job_name']=config['job_name'] 
            else:
              metaconfig['job_name']='gcp_job'

            submit_job(metaconfig,args,package_uris)
        elif FLAGS.package=='local':
            if 'setup_loc' in metaconfig:
              metaconfig['setup_loc']+='/'
            else:
              metaconfig['setup_loc']=''
            script=metaconfig['setup_loc']+'/'.join(metaconfig['module'].split('.'))+'.py'
            # subprocess.call(['python3',metaconfig['setup_loc']+'setup.py','install'])
            subprocess.call(['python3',script]+args)
        elif FLAGS.docker[:5]=='local':
            if not FLAGS.GPU:
              subprocess.call(['docker', 'run', metaconfig['docker_tag']]+args)
            else:
              subprocess.call(['nvidia-docker', 'run', metaconfig['docker_tag']]+args)
        elif FLAGS.docker=='shell':
            
            if FLAGS.GPU:
              subprocess.call(['nvidia-docker', 'run' ,'--entrypoint', '/bin/bash', '-it', metaconfig['docker_tag']])
            else:
              subprocess.call(['docker', 'run' ,'--entrypoint', '/bin/bash', '-it', metaconfig['docker_tag']])




if __name__ == '__main__':
  app.run(main)

