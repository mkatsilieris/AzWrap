import os
import sys
from io import BytesIO
from docx import Document
from dotenv import load_dotenv
import json
from tqdm import tqdm


# === Dynamic Path Setup ===
# Set up the directory paths to ensure proper access to the AzWrap framework
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AZWRAP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(AZWRAP_ROOT)

# === Imports from AzWrap and internal modules ===
# Import necessary Azure wrappers for resource management
from AzWrap.wrapper import (
    Identity, Subscription, ResourceGroup, StorageAccount,
    SearchService, SearchIndex, AIService, OpenAIClient
)

# Import custom modules for document processing and data ingestion
from knowledge_pipeline_from_docx.doc_parsing import DocParsing
from knowledge_pipeline_from_docx.ingestion import MultiProcessHandler

# Load environment variables for configuration
load_dotenv()
CONTAINER_NAME = os.getenv("CONTAINER_NAME")  # Azure Blob Storage container name
TEMP_PATH =  r"C:\Users\agkithko\OneDrive - Netcompany\Desktop\AzWrap-1\temp_json"  # Path for temporary JSON files
FORMAT_JSON_PATH = r"C:\Users\agkithko\OneDrive - Netcompany\Desktop\AzWrap-1\knowledge_pipeline_from_docx\format.json"  # Path to JSON format definition
SEARCH_SERVICE_NAME = os.getenv("SEARCH_SERVICE_NAME")  # Azure Cognitive Search service name
TARGET_ACCOUNT_NAME = os.getenv("TARGET_ACCOUNT_NAME")  # Target Azure account name for AI services
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")  # Azure resource group name
TENANT_ID = os.getenv("AZURE_TENANT_ID")  # Azure tenant ID for authentication
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")  # Azure client ID for authentication
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")  # Azure client secret for authentication
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")  # Azure subscription ID
CORE_INDEX_NAME = os.getenv("CORE_INDEX_NAME")  # Name of the core search index
DETAILED_INDEX_NAME = os.getenv("DETAILED_INDEX_NAME")  # Name of the detailed search index
ACCOUNT_NAME = os.getenv("ACCOUNT_NAME")  # Azure storage account name
MODEL_NAME = os.getenv("MODEL_NAME")  # OpenAI model name to use


def load_format(format_path):
    """
    Loads the JSON format file that defines how documents should be processed.
    
    Args:
        format_path (str): Path to the JSON format file
        
    Returns:
        dict: The loaded JSON format configuration
    """
    print("📂 Loading JSON format file...")
    # Open and read the JSON file
    with open(format_path, "r", encoding="utf-8") as file:
        return json.load(file)

# === Utility Functions ===
def delete_files_in_directory(directory_path):
    """
    Deletes all files in a given directory to clean up temporary files.
    
    Args:
        directory_path (str): Path to the directory to clean
    """
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
    """
    Returns a list of all file paths in a given directory, excluding 'Metadata' files.
    
    Args:
        directory_path (str): Directory to scan for files
        
    Returns:
        list: List of file paths found in the directory
    """
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
    """
    Gets the Azure OpenAI Client for AI operations.
    
    Args:
        sub (Subscription): Azure subscription object
        rg (ResourceGroup): Azure resource group object
        target_account_name (str): Target Azure account name
        
    Returns:
        OpenAIClient: Configured Azure OpenAI client
    """
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
    """
    Initialize all Azure clients needed for the pipeline.
    
    Returns:
        dict: Dictionary containing all initialized client objects
    """
    # Set up Azure authentication with service principal
    identity = Identity(tenant_id=TENANT_ID, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    subscription_info = identity.get_subscription(SUBSCRIPTION_ID)
    sub = Subscription(identity=identity, subscription=subscription_info, subscription_id=subscription_info.subscription_id)

    # Initialize resource group and storage access
    rg = sub.get_resource_group(RESOURCE_GROUP)
    storage_account = rg.get_storage_account(ACCOUNT_NAME)
    print(CONTAINER_NAME)
    container = storage_account.get_container(CONTAINER_NAME)
    folder_structure = container.get_folder_structure()

    # Set up search services and indices
    search_service = sub.get_search_service(SEARCH_SERVICE_NAME)
    core_index = search_service.get_index(CORE_INDEX_NAME)
    detail_index = search_service.get_index(DETAILED_INDEX_NAME)
    
    # Initialize Azure OpenAI client
    azure_oai = get_azure_oai(sub, rg)

    # Return all clients in a dictionary for easy access
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

# Files for tracking processing progress
CHECKPOINT_FILE = "processed_files.txt"  # Tracks successfully processed files
FAILED_FILE_LOG = "failed_files.txt"     # Tracks files that failed processing

def load_processed_files():
    """
    Load the list of previously processed files from checkpoint file.
    
    Returns:
        set: Set of processed file names
    """
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines())

def save_processed_file(filename):
    """
    Save a successfully processed file to the checkpoint file.
    
    Args:
        filename (str): Name of the processed file
    """
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(filename + "\n")

def log_failed_file(filename):
    """
    Log a failed file for retry later.
    
    Args:
        filename (str): Name of the failed file along with error info
    """
    with open(FAILED_FILE_LOG, "a", encoding="utf-8") as f:
        f.write(filename + "\n")

def load_failed_files():
    """
    Load the list of previously failed files.
    
    Returns:
        set: Set of failed file names
    """
    if not os.path.exists(FAILED_FILE_LOG):
        return set()
    with open(FAILED_FILE_LOG, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines())
    

# === Main Logic ===
def main():
    """
    Main function to run the knowledge pipeline process.
    
    The pipeline performs the following steps:
    1. Initialize all Azure clients
    2. Load checkpoint data (processed and failed files)
    3. Process documents from Azure Blob Storage
    4. Parse DOCX files using AI to extract structured information
    5. Upload processed information to Azure Cognitive Search indices
    """
    # Initialize all clients needed for the pipeline
    clients = initialize_clients()
    
    # Load tracking information from checkpoint files
    processed_files = load_processed_files()
    failed_files = load_failed_files()

    # Set up files to process collection
    files_to_process = set()
    
    # Collect files that need processing (not previously processed)
    for folder, files in clients["folder_structure"].items():
        for file in files:
            if file not in processed_files:
                files_to_process.add((folder, file))  # Add as a tuple (folder, file)

    # Process each file in the queue with progress bar
    for folder, file in tqdm(files_to_process):
        # Construct blob path for Azure Storage
        blob_path = f"{folder}/{file}" if folder else file
        
        # Clean temporary directory before processing
        delete_files_in_directory(TEMP_PATH)
        
        try:
            # Download and open the DOCX file from Azure Blob Storage
            print(f"Processing {folder}/{file}...")
            blob = clients["container"].get_blob_content(blob_path)
            byte_stream = BytesIO(blob)
            docx_document = Document(byte_stream)
        except Exception as e:
            # Handle document download errors
            print(f"⚠️ Error reading {blob_path}: {e}")
            log_failed_file(file)  # Log the failed file for retrying later
            continue  # Skip to the next file

        try:
            # Parse document content using AI and extract structured information
            parser = DocParsing(
                doc_instance=docx_document,
                client=clients["azure_oai"],
                json_file=load_format(FORMAT_JSON_PATH),
                domain="-",
                sub_domain=folder,
                model_name=MODEL_NAME,
                doc_name=file.replace('.docx', "")
            )

            # Convert document to structured JSON format
            parser.doc_to_json(doc_name=file.replace('.docx', ""))
            
            # Get all generated JSON files for processing
            json_files = get_all_files_in_directory(TEMP_PATH)
            
            # Process the JSON files in parallel and prepare for indexing
            processor = MultiProcessHandler(json_files, clients["core"], clients["detail"], clients["azure_oai"])

            # Extract records from processed documents
            records = processor.process_documents()
            
            # Upload processed records to Azure Cognitive Search indices
            processor.upload_to_azure_index(records, CORE_INDEX_NAME, DETAILED_INDEX_NAME)
            
            # Mark file as successfully processed
            save_processed_file(file)
            print(f"✅ {file} processed successfully!")

            exit()  # Exit after processing one file (debug mode)

        except Exception as e:
            # Handle processing errors
            print(f"⚠️ Error processing {file}: {e}")
            log_failed_file(f"{file} problem: {e}")  # Log the failed file with error details
            continue  # Skip to the next file


if __name__ == "__main__":
    main()

