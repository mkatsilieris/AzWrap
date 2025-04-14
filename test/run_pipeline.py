#!/usr/bin/env python3
"""
Run Pipeline Script for AzWrap

This script provides a simple way to run the AzWrap pipeline command
directly without using the CLI interface. It's equivalent to running:
python -m AzWrap.main pipeline run
"""

import os
import sys
from dotenv import load_dotenv

# Import the run_pipeline function from AzWrap
from AzWrap.wrapper import initialize_clients, load_format, delete_files_in_directory
from AzWrap.wrapper import load_processed_files, load_failed_files, get_all_files_in_directory
from AzWrap.wrapper import DocParsing, MultiProcessHandler, manage_azure_search_indexes


def main():
    """Execute the AzWrap pipeline run command."""
    print("🚀 Starting AzWrap Pipeline Run")
    
    # Load configuration from environment variables
    load_dotenv()
    
    # Add required config items for the pipeline
    pipeline_config = {
        "AZURE_TENANT_ID": os.getenv("AZURE_TENANT_ID"),
        "AZURE_CLIENT_ID": os.getenv("AZURE_CLIENT_ID"),
        "AZURE_CLIENT_SECRET": os.getenv("AZURE_CLIENT_SECRET"),
        "AZURE_SUBSCRIPTION_ID": os.getenv("AZURE_SUBSCRIPTION_ID"),
        "RESOURCE_GROUP": os.getenv("RESOURCE_GROUP"),
        "TARGET_ACCOUNT_NAME": os.getenv("TARGET_ACCOUNT_NAME"),
        "ACCOUNT_NAME": os.getenv("ACCOUNT_NAME"),
        "CONTAINER_NAME": os.getenv("CONTAINER_NAME"),
        "SEARCH_SERVICE_NAME": os.getenv("SEARCH_SERVICE_NAME"),
        "CORE_INDEX_NAME": os.getenv("CORE_INDEX_NAME"),
        "DETAILED_INDEX_NAME": os.getenv("DETAILED_INDEX_NAME"),
        "MODEL_NAME": os.getenv("MODEL_NAME"),
        "TEMP_PATH": os.getenv("TEMP_PATH", os.path.join(os.getcwd(), "temp_json")),
        "FORMAT_JSON_PATH": os.getenv("FORMAT_JSON_PATH", os.path.join(os.getcwd(), "format.json")),
    }
    
    # Check for required config
    missing = [key for key, value in pipeline_config.items() if not value]
    if missing:
        print(f"❌ Missing required configuration: {', '.join(missing)}")
        print("Set these variables in your .env file")
        return 1
    
    # Set parameters for the pipeline run
    manage_indexes = False  # Set to True if you want to manage indexes
    recreate_indexes = False  # Set to True if you want to recreate indexes
    
    print("\n🔧 Pipeline Configuration:")
    print(f"- Manage Indexes: {manage_indexes}")
    print(f"- Recreate Indexes: {recreate_indexes}")
    
    # Call the run_pipeline function from AzWrap.main
    from AzWrap.main import run_pipeline
    result = run_pipeline(pipeline_config, manage_indexes, recreate_indexes)
    
    print("\n✅ Pipeline execution completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())