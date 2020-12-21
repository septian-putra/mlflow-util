'''Experiment Object is Connector to MLflowClient'''
import os
from git import Repo
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_TYPE, MLFLOW_SOURCE_NAME, MLFLOW_GIT_REPO_URL, \
    MLFLOW_GIT_BRANCH, MLFLOW_GIT_COMMIT


class Experiment(MlflowClient):
    '''The wrapper function to use MlflowClient
    in https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html

    Parameters
    ----------
    conf : dict
        Configuration Description of parameter `x`.

    Attributes
    ----------
    experiment_name: str
        The name of the experiment.
    experiment_id: str
        The id of the experiment.
    tracking_uri: the str
        The uri of tracking server, i.e. http://111.222.123.234:5000
    artifact_location: str
        The uri of artifact location, i.e. s3://exact-mlflow/log
    git_directory:str
        The directory in which .git located.
    run_uuid: str
        The uuid of current active run.
    run_tags: dict
        The dictionary containing the active run's tag
    source_type: str
        'NOTEBOOK', 'JOB', 'PROJECT', 'LOCAL'
    '''

    def __init__(self, conf=None):
        MlflowClient.__init__(self, conf.get('tracking_uri'))
        mlflow.set_tracking_uri(conf.get('tracking_uri'))
        self.artifact_location = conf.get('artifact_location')
        self.user_id = conf.get('user_id')
        self.git_directory = conf.get('use_git_version', None)
        self.experiment_name = conf.get('experiment_name')
        self.experiment_id = None
        self.run_tags = {}
        self.run_uuid = None
        self.source_type = 'NOTEBOOK' if not conf.get(
            'source_type') else conf.get('source_type')

        current_experiment = self.get_experiment_by_name(self.experiment_name)
        if not current_experiment:
            self.experiment_id = self.create_experiment(
                self.experiment_name, self.artifact_location)
        else:
            self.experiment_id = current_experiment.experiment_id
        mlflow.set_experiment(self.experiment_name)

    def create_run(self, run_name=None, run_tags=None):
        '''Create active run

        Parameters
        ----------
        run_name : str
            The name of the run.
        run_tags : dict
            Contains run tag.
        '''
        if self.git_directory:
            repo = Repo(self.git_directory)
#            assert not repo.is_dirty()
            self.run_tags[MLFLOW_SOURCE_TYPE] = self.source_type
            self.run_tags[MLFLOW_SOURCE_NAME] = repo.remotes[-1].url
            self.run_tags[MLFLOW_GIT_REPO_URL] = repo.remotes[-1].url
            self.run_tags[MLFLOW_GIT_BRANCH] = repo.branches[-1].name
            self.run_tags[MLFLOW_GIT_COMMIT] = repo.head.commit.hexsha
        self.run_tags['mlflow.user'] = self.user_id
        self.run_tags['mlflow.runName'] = run_name
        self.run_uuid = super(Experiment, self).create_run(
            self.experiment_id, tags=self.run_tags).info.run_uuid
        mlflow.start_run(self.run_uuid, self.experiment_id)

    def log_artifact(self, local_path, artifact_path):
        '''Send artifact to artifact_location

        Parameters
        ----------
        local_path : str
            The local path of artifact file.
        artifact_path : str
            The subdirectory under artifact_location.
        '''
        super(Experiment, self).log_artifact(
            self.run_uuid, local_path, artifact_path)

    def log_artifacts(self, local_dir, artifact_path):
        '''Send artifacts to artifact_location

        Parameters
        ----------
        local_dir : str
            The local path of artifact directory.
        artifact_path : str
            The subdirectory under artifact_location.
        '''
        super(Experiment, self).log_artifacts(
            self.run_uuid, local_dir, artifact_path)

    def log_params(self, hyperparam_d):
        '''Send parameter to tracking_uri

        Parameters
        ----------
        hyperparam_d : dict
            The dictionary of hyperparameter.
        '''
        for key, value in hyperparam_d.items():
            self.log_param(self.run_uuid, key, value)

    def log_metrics(self, metric_d):
        '''Send metric to tracking_uri

        Parameters
        ----------
        metric_d : dict
            The dictionary of metric, the value must be a numerical or a list of numerical.
        '''
        for key, value in metric_d.items():
            if isinstance(value, list):
                for val in value:
                    self.log_metric(self.run_uuid, key, val)
            else:
                self.log_metric(self.run_uuid, key, value)

    def terminate_run(self):
        '''Terminate current active run
        '''
        self.run_uuid = None
        mlflow.end_run()


def log_experiment(run_name, hyperparam, metric, artifact_dir, from_ec2=False):
    if not os.path.isdir(artifact_dir):
        raise Exception("Artifacts directory not found!")
    config = {
        'experiment_name': 'purchase-invoice',
        'user_id': "sagemaker",
        # need to open port when WFH
        'tracking_uri': 'http://10.1.2.157:5000' if from_ec2 else 'http://52.19.172.239:5000',
        'artifact_location': 's3://cig-ds-dev/mlflow',
        'git_directory': None
    }
    # Initiate the experiment and run entities
    ex_log = Experiment(config)
    ex_log.create_run(run_name)

    # Save the hyperparam & metric
    ex_log.log_params(hyperparam)
    ex_log.log_metrics(metric)

    # Save artifacts (model, vectorizer, etc)
    ex_log.log_artifacts(artifact_dir, 'artifacts')

    # Terminate run
    ex_log.terminate_run()
