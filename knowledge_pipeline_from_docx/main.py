import os
import sys
from io import BytesIO
from docx import Document
from dotenv import load_dotenv
import json

# === Dynamic Path Setup ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AZWRAP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(AZWRAP_ROOT)

# === Imports from AzWrap and internal modules ===
from AzWrap.wrapper import (
    Identity, Subscription, ResourceGroup, StorageAccount,
    SearchService, SearchIndex, AIService, OpenAIClient
)

from knowledge_pipeline_from_docx.doc_parsing import DocParsing
from knowledge_pipeline_from_docx.upload import MultiProcessHandler

load_dotenv()
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
TEMP_PATH =  r"C:\Users\agkithko\OneDrive - Netcompany\Desktop\nbg_wrap\AzWrap\temp_json"
FORMAT_JSON_PATH = r"C:\Users\agkithko\OneDrive - Netcompany\Desktop\nbg_wrap\AzWrap\knowledge_pipeline\format.json"
SEARCH_SERVICE_NAME = os.getenv("SEARCH_SERVICE_NAME")
TARGET_ACCOUNT_NAME = os.getenv("TARGET_ACCOUNT_NAME")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
CORE_INDEX_NAME = os.getenv("CORE_INDEX_NAME")
DETAILED_INDEX_NAME = os.getenv("DETAILED_INDEX_NAME")
ACCOUNT_NAME = os.getenv("ACCOUNT_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")


def load_format(format_path):
    print("📂 Loading JSON format file...")
    # Open and read the JSON file
    with open(format_path, "r", encoding="utf-8") as file:
        return json.load(file)


# === Utility Functions ===
def delete_files_in_directory(directory_path):
    """Deletes all files in a given directory."""
    print(f"🗑️ Starting file deletion in directory: {directory_path}")
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"   ✔️ Deleted: {file_path}")
    else:
        print(f"❌ Directory '{directory_path}' does not exist or is not a directory.")
    print("🏁 File deletion process completed")

def get_all_files_in_directory(directory_path):
    """Returns a list of all file paths in a given directory, excluding 'Metadata' files."""
    print(f"📂 Scanning directory for files: {directory_path}")
    file_list = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if "Metadata" not in file:
                file_path = os.path.join(root, file)
                file_list.append(file_path)
                print(f"   📄 Found file: {file_path}")
    print(f"🔢 Total files found: {len(file_list)}")
    return file_list

def get_azure_oai(sub, rg, target_account_name=TARGET_ACCOUNT_NAME):
    """Gets the Azure OpenAI Client."""
    cog = sub.get_cognitive_client()
    account = next(
        (acc for acc in cog.accounts.list_by_resource_group(rg.azure_resource_group.name)
         if acc.name == target_account_name),
        None
    )

    if account:
        print(f"✅ Selected Account: {account.name}, Location: {account.location}")
        ai = AIService(rg, cog, account)
        return ai.get_AzureOpenAIClient(api_version='2024-05-01-preview')
    else:
        print("❌ Account not found.")
        return None
    
def initialize_clients():
    identity = Identity(tenant_id=TENANT_ID, client_id=CLIENT_ID , client_secret=CLIENT_SECRET)
    subscription_info = identity.get_subscription(SUBSCRIPTION_ID)
    sub = Subscription(identity=identity, subscription=subscription_info, subscription_id=subscription_info.subscription_id)

    rg = sub.get_resource_group(RESOURCE_GROUP)
    storage_account = rg.get_storage_account(ACCOUNT_NAME)
    print(CONTAINER_NAME)
    container = storage_account.get_container(CONTAINER_NAME)
    folder_structure = container.get_folder_structure()

    search_service = sub.get_search_service(SEARCH_SERVICE_NAME)
    core_index = search_service.get_index(CORE_INDEX_NAME)
    detail_index = search_service.get_index(DETAILED_INDEX_NAME)
    azure_oai = get_azure_oai(sub, rg)

    return {
        "sub": sub,
        "rg": rg,
        "container": container,
        "folder_structure": folder_structure,
        "core": core_index.get_search_client(),
        "detail": detail_index.get_search_client(),
        "azure_oai": azure_oai
    }

import os

import os

CHECKPOINT_FILE = "processed_files.txt"
FAILED_FILE_LOG = "failed_files.txt"

# Load the list of processed files
def load_processed_files():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines())

# Save processed files to a checkpoint file
def save_processed_file(filename):
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(filename + "\n")

# Log failed files for retry later
def log_failed_file(filename):
    with open(FAILED_FILE_LOG, "a", encoding="utf-8") as f:
        f.write(filename + "\n")

# Load the list of failed files
def load_failed_files():
    if not os.path.exists(FAILED_FILE_LOG):
        return set()
    with open(FAILED_FILE_LOG, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines())

# === Main Logic ===
def main():
    clients = initialize_clients()
    processed_files = load_processed_files()
    failed_files = load_failed_files()

    # Attempt to process the backlog first (if any)
    files_to_process = set()
    
    # If there are failed files, try to process them first
    files_to_process.update(failed_files)
    
    # Correct way to add files_to_process as tuples of (folder, file)
    files_to_process = set()

    # If there are failed files, try to process them first
    for folder, files in clients["folder_structure"].items():
        for file in files:
            if file not in processed_files and file not in failed_files:
                files_to_process.add((folder, file))  # Add as a tuple (folder, file)

    # Then process the files_to_process
    for folder, file in files_to_process:
        blob_path = f"{folder}/{file}" if folder else file
        try:
            print(f"Processing {folder}/{file}...")
            blob = clients["container"].get_blob_content(blob_path)
            byte_stream = BytesIO(blob)
            docx_document = Document(byte_stream)
        except Exception as e:
            print(f"⚠️ Error reading {blob_path}: {e}")
            log_failed_file(file)  # Log the failed file for retrying later
            continue  # Skip to the next file

        try:
            # Processing the document
            parser = DocParsing(
                doc_instance=docx_document,
                client=clients["azure_oai"],
                json_file=load_format(FORMAT_JSON_PATH),
                domain="-",
                sub_domain=folder,
                model_name=MODEL_NAME,
                doc_name=file.replace('.docx', "")
            )
            parser.doc_to_json(doc_name=file.replace('.docx', ""))

            json_files = get_all_files_in_directory(TEMP_PATH)
            processor = MultiProcessHandler(json_files, clients["core"], clients["detail"], clients["azure_oai"])

            records = processor.process_documents()
            processor.upload_to_azure_index(records, CORE_INDEX_NAME, DETAILED_INDEX_NAME)
            delete_files_in_directory(TEMP_PATH)

            # Successfully processed, save it to checkpoint
            save_processed_file(file)
            print(f"✅ {file} processed successfully!")

        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")
            log_failed_file(file)  # Log the failed file
            log_failed_file(e)
            continue  # Skip to the next file


if __name__ == "__main__":
    main()

