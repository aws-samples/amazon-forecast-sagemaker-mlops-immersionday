# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Example workflow pipeline script for CustomerChurn pipeline.
                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)
Implements a get_pipeline(**kwargs) method.
"""

import os

import boto3
import sagemaker
import sagemaker.session

import json
import tarfile
import time

import botocore
import zipfile
import pandas as pd

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import (
    ConditionGreaterThanOrEqualTo,
)
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel


from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ( 
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.s3 import S3Downloader, S3Uploader


BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AmazonForecastPackageGroup",  # Choose any name
    pipeline_name="CustomAmazonForecastMLOps-p-m8labnjdek1j",  # You can find your pipeline name in the Studio UI (project -> Pipelines -> name)
    base_job_prefix="AmazonForecast",  # Choose any name
):
    """Gets a SageMaker ML Pipeline instance working with on Amazonforecast data.
    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # Parameters for pipeline execution
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.large"
    )
    training_instance_count = ParameterInteger(
        name="TrainingInstanceCount", default_value=1
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.large"
    ) 
    input_train = ParameterString(
        name="TrainData",
        default_value=f"s3://sagemaker-eu-west-1-870401269756/amazon-forecast-mlops/data/train.csv",
    )
    model_output = ParameterString(name="ModelOutput", default_value=f"s3://{default_bucket}/model")

    # Model parameters
    forecast_horizon = ParameterString(
        name="ForecastHorizon", default_value="24"
    )
    forecast_algorithm = ParameterString(
        name="ForecastAlgorithm", default_value="NPTS"
    )
    maximum_score = ParameterString(
        name="MaxScore", default_value="0.4"
    )
    metric = ParameterString(
        name="EvaluationMetric", default_value="WAPE"
    )

    image_uri = sagemaker.image_uris.retrieve(
        framework="sklearn",  # we are using the Sagemaker built in xgboost algorithm
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.c5.xlarge",
    )
    
    sklearn_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name="forecast-process",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    preprocess = ProcessingStep(
        name="ForecastPreProcess",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=input_train, destination="/opt/ml/processing/input_train"),
        ],
        outputs=[
            ProcessingOutput(output_name="target", source="/opt/ml/processing/target"),
            ProcessingOutput(output_name="related", source="/opt/ml/processing/related"),
        ],
        job_arguments=["--forecast_horizon", forecast_horizon],
        code=os.path.join(BASE_DIR, "preprocess.py"),
    )

    # Training step for generating model artifacts
#    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/AmazonForecastTrain"

    # Define the hyperparmeters and the Regex Syntax associated to the metrics
    hyperparameters = {
        "forecast_horizon": forecast_horizon,
        "forecast_algorithm": forecast_algorithm,
        "dataset_frequency": "H",
        "timestamp_format": "yyyy-MM-dd hh:mm:ss",
        "number_of_backtest_windows": "1",
        "s3_directory_target": preprocess.properties.ProcessingOutputConfig.Outputs[
            "target"
        ].S3Output.S3Uri,
        "s3_directory_related": preprocess.properties.ProcessingOutputConfig.Outputs[
            "related"
        ].S3Output.S3Uri,
        "role_arn": role,
        "region": region,
    }
    metric_definitions = [
        {"Name": "WAPE", "Regex": "WAPE=(.*?);"},
        {"Name": "RMSE", "Regex": "RMSE=(.*?);"},
        {"Name": "MASE", "Regex": "MASE=(.*?);"},
        {"Name": "MAPE", "Regex": "MAPE=(.*?);"},
    ]

    forecast_model = SKLearn(
        entry_point=os.path.join(BASE_DIR, "train.py"),
        role=role,
        image_uri=image_uri,
        #instance_count=training_instance_count,
        instance_type=training_instance_type,
        sagemaker_session=sagemaker_session,
        base_job_name="forecast-train",
        hyperparameters=hyperparameters,
        enable_sagemaker_metrics=True,
        metric_definitions=metric_definitions,
    )
    

    forecast_train_and_eval = TrainingStep(
        name="ForecastTrainAndEvaluate", estimator=forecast_model
    )

    postprocess = ProcessingStep(
        name="ForecastCondtionalDelete",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=forecast_train_and_eval.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
        ],
        job_arguments=[
            "--metric",
            metric,
            "--maximum-score",
            maximum_score,
            "--region",
            region,
        ],
        code=os.path.join(BASE_DIR, "conditional_delete.py"),

    )

    #pipeline_name = "CustomForecastPipeline"
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_count,
            training_instance_type,
            input_train,
            forecast_horizon,
            forecast_algorithm,
            model_output,
            metric,
            maximum_score,
        ],
        steps=[preprocess, forecast_train_and_eval, postprocess],
        pipeline_experiment_config=PipelineExperimentConfig(
            ExecutionVariables.PIPELINE_NAME,
            Join(on="-", values=["ForecastTrial", ExecutionVariables.PIPELINE_EXECUTION_ID]),
        ),
    )
    return pipeline
