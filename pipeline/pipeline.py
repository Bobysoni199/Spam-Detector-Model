

import os
import subprocess
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.entities import (
    Model,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration
)
from azure.ai.ml.entities._job.job_resource_configuration import JobResourceConfiguration
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# ================================================================
# CONFIG
# ================================================================

SUBSCRIPTION_ID = "31071f64-adbc-4ffa-93de-8c69ac43184a"
RESOURCE_GROUP  = "RnD-Bhavishya-RG"
WORKSPACE_NAME  = "azure-ml199"
ENDPOINT_NAME   = "spam-detector-endpoint"
DEPLOYMENT_NAME = "spam-deployment"
ENVIRONMENT     = "azureml://registries/azureml/environments/sklearn-1.5/versions/40"
ROOT            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================================================================
# RETRAIN CONFIG
# ← ONLY CHANGE THESE TO RETRAIN WITH NEW DATA!
# ================================================================

NORMAL_COUNT = 1500   # number of normal transactions
SPAM_COUNT   = 500  # number of spam transactions

# ================================================================
# CONNECT TO AZURE ML
# ================================================================

print("=" * 60)
print("Connecting to Azure ML...")
credential = DefaultAzureCredential()
ml_client  = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME
)
print("✅ Connected!")
print(f"Retraining with {NORMAL_COUNT} normal + {SPAM_COUNT} spam transactions")

# ================================================================
# STEP 1: PREPROCESS
# runs preprocess.py ON AZURE
# passes NORMAL_COUNT + SPAM_COUNT as arguments
# output → saved to Azure Blob automatically
# visible in Azure ML Studio → Jobs
# ================================================================

preprocess_component = command(
    name="preprocess",
    display_name="Step 1: Preprocess Data",
    description=f"Generates {NORMAL_COUNT} normal + {SPAM_COUNT} spam transactions",
    command=(
        f"python preprocess.py "
        f"--output_data ${{{{outputs.output_data}}}} "
        f"--normal_count {NORMAL_COUNT} "
        f"--spam_count {SPAM_COUNT}"
    ),
    code=f"{ROOT}/src",
    environment=ENVIRONMENT,
    compute="serverless",
    resources=JobResourceConfiguration(
        instance_type="Standard_E2s_v3",
        instance_count=1
    ),
    outputs={
        "output_data": Output(
            type=AssetTypes.URI_FOLDER,
            path="azureml://datastores/workspaceblobstore/paths/spam-detector/data/"
        )
    }
)

# ================================================================
# STEP 2: TRAIN
# runs train.py ON AZURE
# reads output of Step 1
# logs metrics to MLflow → visible in Azure ML Studio
# output → saved to Azure Blob automatically
# ================================================================

train_component = command(
    name="train",
    display_name="Step 2: Train Model",
    description="Trains model + logs metrics to MLflow",
    command=(
        "python train.py "
        "--input_data ${{inputs.input_data}} "
        "--output_model ${{outputs.output_model}} "
        "--output_metrics ${{outputs.output_metrics}}"
    ),
    code=f"{ROOT}/src",
    environment=ENVIRONMENT,
    compute="serverless",
    resources=JobResourceConfiguration(
        instance_type="Standard_E2s_v3",
        instance_count=1
    ),
    inputs={
        "input_data": Input(type=AssetTypes.URI_FOLDER)
    },
    outputs={
        "output_model": Output(
            type=AssetTypes.URI_FOLDER,
            path="azureml://datastores/workspaceblobstore/paths/spam-detector/models/"
        ),
        "output_metrics": Output(
            type=AssetTypes.URI_FOLDER,
            path="azureml://datastores/workspaceblobstore/paths/spam-detector/metrics/"
        )
    }
)

# ================================================================
# BUILD PIPELINE
# ================================================================

@pipeline(
    name="spam-detector-pipeline",
    description="Preprocess → Train → visible in Azure ML Studio"
)
def spam_pipeline():
    # step 1: preprocess
    preprocess_job = preprocess_component()

    # step 2: train → takes output of preprocess as input
    train_job = train_component(
        input_data=preprocess_job.outputs.output_data
    )

    return {
        "clean_data":   preprocess_job.outputs.output_data,
        "model_output": train_job.outputs.output_model,
        "metrics":      train_job.outputs.output_metrics
    }

# ================================================================
# SUBMIT PIPELINE TO AZURE ML
# visible in Azure ML Studio → Jobs → Pipelines
# ================================================================

print("\nSubmitting pipeline to Azure ML...")
pipeline_job  = spam_pipeline()
submitted_job = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="spam-detector-experiment"
)

print(f"✅ Pipeline submitted!")
print(f"Job name : {submitted_job.name}")
print(f"Status   : {submitted_job.status}")
print(f"\n👉 View pipeline in Azure ML Studio:")
print(f"https://ml.azure.com/runs/{submitted_job.name}?wsid=/subscriptions/{SUBSCRIPTION_ID}/resourcegroups/{RESOURCE_GROUP}/workspaces/{WORKSPACE_NAME}")

# wait for pipeline to complete on Azure
print("\nWaiting for pipeline to finish on Azure...")
print("(You can also watch it in Azure ML Studio!)")
ml_client.jobs.stream(submitted_job.name)
print("✅ Pipeline complete!")

# ================================================================
# DVC - VERSION NEW DATA + MODELS IN AZURE BLOB
# files were saved to Blob by Azure ML Pipeline
# DVC now pulls them → records new version → pushes back
# this means we can go back to any version anytime!
# ================================================================

print("\nDVC: Pulling new files from Azure Blob...")
subprocess.run(["dvc", "pull"], cwd=ROOT)

print("\nDVC: Committing new versions...")
# dvc commit = file already tracked → just update hash
# input=b"y\n" = auto answer yes to "file changed, sure? [y/n]"
subprocess.run(["dvc", "commit", "data/clean_data.csv"],   cwd=ROOT, input=b"y\n")
subprocess.run(["dvc", "commit", "models/spam_model.pkl"], cwd=ROOT, input=b"y\n")
subprocess.run(["dvc", "commit", "models/scaler.pkl"],     cwd=ROOT, input=b"y\n")

print("\nDVC: Pushing new versions to Azure Blob...")
subprocess.run(["dvc", "push"], cwd=ROOT, check=True)
print("✅ DVC versioning complete!")

# ================================================================
# REGISTER MODEL FROM PIPELINE OUTPUT
# registers from Azure Blob output path
# version auto increments: 4 → 5 → 6...
# visible in Azure ML Studio → Models
# ================================================================

print("\nRegistering model from pipeline output...")
model_entity = Model(
    path=f"azureml://jobs/{submitted_job.name}/outputs/model_output",
    name="spam-transaction-model",
    description=f"Spam detector - {NORMAL_COUNT} normal + {SPAM_COUNT} spam transactions",
    type="custom_model"
)
registered_model = ml_client.models.create_or_update(model_entity)
print(f"✅ Model registered: version {registered_model.version}")
print(f"👉 View: Azure ML Studio → Models → spam-transaction-model")

# ================================================================
# CREATE ENDPOINT IF NOT EXISTS
# visible in Azure ML Studio → Endpoints
# ================================================================

print("\nChecking endpoint...")
try:
    ml_client.online_endpoints.get(ENDPOINT_NAME)
    print("✅ Endpoint already exists!")
except Exception:
    print("Creating endpoint...")
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Spam transaction detector endpoint",
        auth_mode="key"
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print("✅ Endpoint created!")

# ================================================================
# RETRAIN + REDEPLOY
# checks if old deployment exists → deletes it → deploys new!
# handles everything automatically!
# ================================================================

print("\nChecking existing deployment...")
try:
    # check if old deployment exists
    ml_client.online_deployments.get(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME
    )
    print("Old deployment found! Replacing with new model...")

    # step 1: set traffic to 0 first
    # Azure ML requires traffic = 0 before deleting!
    print("Setting traffic to 0...")
    endpoint         = ml_client.online_endpoints.get(ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 0}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print("✅ Traffic set to 0!")

    # step 2: delete old deployment
    print("Deleting old deployment...")
    ml_client.online_deployments.begin_delete(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME
    ).result()
    print("✅ Old deployment deleted!")

except Exception:
    # no old deployment → fresh deploy!
    print("No existing deployment → creating fresh!")

# step 3: deploy new model
print("\nDeploying new model...")
deployment = ManagedOnlineDeployment(
    name=DEPLOYMENT_NAME,
    endpoint_name=ENDPOINT_NAME,
    model=registered_model.id,       # new model version!
    instance_type="Standard_E2s_v3",
    instance_count=1,
    code_configuration=CodeConfiguration(
        code=f"{ROOT}/src",
        scoring_script="score.py"
    ),
    environment=ENVIRONMENT
)
ml_client.online_deployments.begin_create_or_update(deployment).result()
print("✅ New model deployed!")

# ================================================================
# SET TRAFFIC TO 100%
# ================================================================

print("\nSetting traffic to 100%...")
endpoint         = ml_client.online_endpoints.get(ENDPOINT_NAME)
endpoint.traffic = {DEPLOYMENT_NAME: 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("✅ Traffic set!")

# ================================================================
# GET URL + KEY
# ================================================================

keys = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)

print("\n" + "=" * 60)
print("✅ RETRAIN + REDEPLOY COMPLETE!")
print("=" * 60)
print(f"Data used         : {NORMAL_COUNT} normal + {SPAM_COUNT} spam")
print(f"New model version : {registered_model.version}")
print(f"Endpoint URL      : {endpoint.scoring_uri}")
print(f"API Key           : {keys.primary_key}")
print(f"""
To retrain again:
→ change NORMAL_COUNT and SPAM_COUNT at top of this file
→ run: python3 pipeline/pipeline.py

View in Azure ML Studio:
👉 Pipeline  : https://ml.azure.com
👉 Experiment: spam-detector-experiment
👉 Model     : spam-transaction-model v{registered_model.version}
👉 Endpoint  : {ENDPOINT_NAME}

Test API:
import requests
response = requests.post(
    "{endpoint.scoring_uri}",
    headers={{
        "Authorization": "Bearer {keys.primary_key}",
        "Content-Type": "application/json",
        "azureml-model-deployment": "{DEPLOYMENT_NAME}"
    }},
    json={{"data": [[4500, 35]]}}
)
print(response.json())
""")