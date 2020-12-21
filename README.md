# mlflow-util
## Introduction
The project is about machine learning experiment logs using MLFlow.
The repository contains wrapper function to use [**mlflow.tracking.MlflowClient**](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html) object. Using this library, a user can log all of their experiment information to a centralized tracking server. The tracking server public access is http://34.243.161.35:5000 and can only be accesed from Exact IP address.

## Server Setup
* Choose the instance type based on the amount of expected requests. I use an `t2.micro` as our team is small. 
* Place the EC2 in public Subnet and create a security group to allow the access to port 5000.
* Assign an EC2 role that allow full S3 Access from EC2 instance for artifacts storage.
* Install mlflow and gitpython package on your python environment.
* Start the MLFLow server by executing the following command:
>```nohup mlflow server --default-artifact-root s3://<artifacts-store-bucket>/mlflow/ --host 0.0.0.0 &``` 

## Client Setup
* Install same mlflow and gitpython package as your MLFlow server.
* Setup the AWS CLI with a proper EC2 role for sending the artifact to AWS S3.
* On EC2:
   - Configure your EC2 to use security group allowed to the access to port 5000 of MLFlow server.
   - During the initialization of Experiment object, configure the `tracking_uri` with MLFlow server's private ipv4 address.
* On machine inside Exact IP address range:
   - During the initialization of Experiment object, configure the `tracking_uri` with MLFlow server's public ipv4 address.

## Quickstart
Initiate the experiment and run
```
import os
from experiment import Experiment
config = {
            'experiment_name': 'mlflow_test',
            'user_id': 'Septian',
            'tracking_uri': 'http://10.1.3.49:5000',
            'artifact_location': 's3://cig-ds-dev/mlflow',
            'use_git_version': True
         }
model_path = os.path.expanduser('~/model')
hyperparam = {}
metric = {}
ex_log = Experiment(config)
ex_log.create_run('sample_run')

# your experment code begin here
```

Save the hyperparam & metric
```
# end of your experment code

ex_log.log_params(hyperparam)
ex_log.log_metrics(metric)
```

Delete or clean model directory
```
shutil.rmtree(model_path)
if not os.path.isdir(model_path): os.makedirs(model_path)
```

Save the model/artifact
```
joblib.dump(model, os.path.join(model_path, 'model.pkl'))
ex_log.log_artifacts(model_path,'model')
```

Terminate run
```
ex_log.terminate_run()
```

## Add this repo as a submodule
```
git submodule add -b master git@github.exactsoftware.com:cloud-solutions/dsci-mlflow-log.git dsci_mlflow_log
```

## Additional link
* More methods for this Experiment object based on [**mlflow.tracking.MlflowClient**](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html)
* [Scaling Ride-Hailing with Machine Learning on MLflow](https://databricks.com/session/scaling-ride-hailing-with-machine-learning-on-mlflow)
