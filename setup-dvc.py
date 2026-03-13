
import subprocess
from azure.identity import DefaultAzureCredential
from azure.mgmt.storage import StorageManagementClient

SUBSCRIPTION_ID      = "d325bbcc-9cfc-40a7-8a4f-93290b27229d"
RESOURCE_GROUP       = "RnD-Bhavishya-RG"
STORAGE_ACCOUNT_NAME = "spamdetectorstorage"
CONTAINER_NAME       = "spam-detector-data"

# ================================================================
# GET STORAGE KEY
# ================================================================

print("Getting storage key...")
credential     = DefaultAzureCredential()
storage_client = StorageManagementClient(credential, SUBSCRIPTION_ID)
keys           = storage_client.storage_accounts.list_keys(
    RESOURCE_GROUP,
    STORAGE_ACCOUNT_NAME
)
storage_key = keys.keys[0].value
print("✅ Storage key retrieved!")

# ================================================================
# INIT DVC
# ================================================================

print("\nInitializing DVC...")
subprocess.run(["dvc", "init"], check=True)
print("✅ DVC initialized!")

# ================================================================
# CONNECT DVC TO AZURE BLOB
# ================================================================

print("\nConnecting DVC to Azure Blob...")
subprocess.run(["dvc", "remote", "add", "-d", "azureblob",
    f"azure://{CONTAINER_NAME}/dvc-store"], check=True)
subprocess.run(["dvc", "remote", "modify", "azureblob",
    "account_name", STORAGE_ACCOUNT_NAME], check=True)
subprocess.run(["dvc", "remote", "modify", "azureblob",
    "account_key", storage_key], check=True)

print("✅ DVC connected to Azure Blob!")
subprocess.run(["dvc", "remote", "list"], check=True)
print("\n✅ Setup complete! Now run: python3 pipeline/pipeline.py")
