#!/usr/bin/env python3
"""
Enhanced entry point for the AzWrap CLI tool with extended functionality.
This module integrates the standard CLI functionality from main.py with
additional features from wrapper_extended.py.
"""

import os
import sys
import json
import logging
import click
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, Any, List, Optional, Union

from dotenv import load_dotenv
import azure.search.documents.indexes.models as azsdim


# Import from wrapper_extended if available
try:
    # Import additional classes or functions from wrapper_extended
    from .wrapper import (
        Identity, Subscription, ResourceGroup, StorageAccount, Container, BlobType,
        SearchService, SearchIndex, SearchIndexerManager, DataSourceConnection, 
        Indexer, Skillset, AIService, OpenAIClient, get_std_vector_search,
        ProcessHandler, DocParsing, MultiProcessHandler
    )
except ImportError:
    # Log warning but continue - extended functionality will be limited
    logging.warning("wrapper_extended module not found. Extended functionality will not be available.")

from .cli_config import CLI_CONFIG, EXTENDED_CLI_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Files for tracking processing progress
CHECKPOINT_FILE = "processed_files.txt"  # Tracks successfully processed files
FAILED_FILE_LOG = "failed_files.txt"     # Tracks files that failed processing

# Global context object to store common objects and settings
class Context:
    def __init__(self):
        self.verbose = False
        self.quiet = False
        self.output_format = "text"
        self.identity = None
        self.subscription = None
        
    def log(self, message, level="info"):
        """Log a message based on verbose/quiet settings."""
        if self.quiet and level != "error":
            return
            
        if level == "debug" and not self.verbose:
            return
            
        if level == "error":
            click.secho(f"ERROR: {message}", fg="red", err=True)
        elif level == "warning":
            click.secho(f"WARNING: {message}", fg="yellow")
        elif level == "success":
            click.secho(message, fg="green")
        elif level == "debug":
            click.secho(f"DEBUG: {message}", fg="cyan")
        else:
            click.echo(message)

    def output(self, data, title=None):
        """Output data in the specified format."""
        if self.output_format == "json":
            click.echo(json.dumps(data, indent=2, default=str))
        elif self.output_format == "table":
            if isinstance(data, list) and data:
                # Try to create a table
                if title:
                    click.secho(title, fg="blue", bold=True)
                    
                # Extract column names from first item
                if isinstance(data[0], dict):
                    headers = list(data[0].keys())
                    rows = [[str(item.get(h, "")) for h in headers] for item in data]
                    
                    # Print the table header
                    header_row = " | ".join(headers)
                    click.echo(header_row)
                    click.echo("-" * len(header_row))
                    
                    # Print table rows
                    for row in rows:
                        click.echo(" | ".join(row))
                else:
                    # Can't create a proper table, fall back to text
                    for item in data:
                        click.echo(str(item))
            else:
                # Output as text for non-list data
                if title:
                    click.secho(title, fg="blue", bold=True)
                click.echo(str(data))
        else:
            # Default to text output
            if title:
                click.secho(title, fg="blue", bold=True)
                
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            click.echo(f"{key}: {value}")
                        click.echo("---")
                    else:
                        click.echo(str(item))
            elif isinstance(data, dict):
                for key, value in data.items():
                    click.echo(f"{key}: {value}")
            else:
                click.echo(str(data))

pass_context = click.make_pass_decorator(Context, ensure=True)

def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    
    # Check if required Azure credentials are available
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    
    missing = []
    if not tenant_id:
        missing.append("AZURE_TENANT_ID")
    if not client_id:
        missing.append("AZURE_CLIENT_ID")
    if not client_secret:
        missing.append("AZURE_CLIENT_SECRET")
        
    if missing:
        click.secho(f"Error: Missing required environment variables: {', '.join(missing)}", fg="red", err=True)
        click.echo("Please set these variables in your .env file or environment.")
        return None
        
    return {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "client_secret": client_secret,
        "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID")
    }

def process_json_arg(json_arg: str) -> Dict:
    """
    Process a JSON argument which could be a string or a file path.
    
    Args:
        json_arg: JSON string or path to JSON file
        
    Returns:
        Parsed JSON object
    """
    if os.path.isfile(json_arg):
        with open(json_arg, 'r') as f:
            return json.load(f)
    else:
        try:
            return json.loads(json_arg)
        except json.JSONDecodeError:
            raise click.BadParameter(f"Invalid JSON format: {json_arg}")

def create_cli():
    """Create the CLI structure from configuration."""
    
    # Create the main CLI group
    @click.group(name=CLI_CONFIG["name"])
    @click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
    @click.option("--quiet", "-q", is_flag=True, help="Suppress all output except errors and results")
    @click.option("--output", "-o", type=click.Choice(["text", "json", "table"]), default="text",
                  help="Output format (text, json, table)")
    @click.option("--config", type=click.Path(exists=True), help="Path to configuration file")
    @click.pass_context
    def cli(ctx, verbose, quiet, output, config):
        """Azure Wrapper (AzWrap) CLI tool for managing Azure resources with extended features."""
        # Initialize the context object
        ctx.obj = Context()
        ctx.obj.verbose = verbose
        ctx.obj.quiet = quiet
        ctx.obj.output_format = output
        
        # Load environment variables
        ctx.obj.log("Loading environment variables", level="debug")
        env = load_environment()
        if not env:
            ctx.obj.log("Failed to load environment variables", level="error")
            sys.exit(1)
            
        # Create identity object
        try:
            ctx.obj.log("Creating identity object", level="debug")
            ctx.obj.identity = Identity(env["tenant_id"], env["client_id"], env["client_secret"])
            
            # If subscription ID is available, initialize subscription
            if env["subscription_id"]:
                ctx.obj.log(f"Initializing subscription with ID: {env['subscription_id']}", level="debug")
                ctx.obj.subscription = ctx.obj.identity.get_subscription(env["subscription_id"])
                
        except Exception as e:
            logger.error(f"Error initializing Azure credentials: {str(e)}")
            click.secho(f"Error initializing Azure credentials: {str(e)}", fg="red", err=True)
            sys.exit(1)
            
        ctx.obj.log("Azure Wrapper CLI initialized", level="debug")
    
    # Add command groups based on the configuration
    all_configs = {**CLI_CONFIG["commands"], **(EXTENDED_CLI_CONFIG.get("commands", {}))}
    for group_name, group_config in all_configs.items():
        group = click.Group(name=group_name, help=group_config["description"])
        cli.add_command(group)
        
        # Add subcommands to the group
        for cmd_name, cmd_config in group_config["subcommands"].items():
            # Create command function
            command_func = create_command_function(group_name, cmd_name, cmd_config)
            
            # Add options to the command
            for option in cmd_config["options"]:
                # Process option attributes
                params = {
                    "help": option.get("help", ""),
                    "required": option.get("required", False),
                    "default": option.get("default", None),
                    "type": click.STRING,  # Default to string type for all params
                }
                
                # Handle flag options
                if option.get("is_flag", False):
                    params["is_flag"] = True
                    params.pop("type")  # Remove type for flag options
                
                # Create the option
                option_name = f"--{option['name']}"
                short_opt = option.get("short")
                if short_opt:
                    # Use the correct format for Click options with short flag
                    command_func = click.option(f"-{short_opt}", option_name, **params)(command_func)
                    continue
                    
                command_func = click.option(option_name, **params)(command_func)
            
            # Add the command to the group
            command = click.command(name=cmd_name, help=cmd_config["description"])(command_func)
            group.add_command(command)
    
    return cli

def create_command_function(group_name, cmd_name, cmd_config):
    """
    Create a command function based on group and command names.
    
    This function dynamically creates CLI command implementations based on the group and command name.
    
    Args:
        group_name: The command group name (e.g., "subscription")
        cmd_name: The specific command name (e.g., "list", "get")
        cmd_config: The command configuration from cli_config.py
        
    Returns:
        A Click command function that implements the specified command
    """
    
    # Generate a function name
    func_name = f"{group_name}_{cmd_name}"
    
    # Define implementation functions based on command
    
    # Handle subscription list command
    if group_name == "subscription" and cmd_name == "list":
        @pass_context
        def func(ctx):
            """List available Azure subscriptions."""
            try:
                ctx.log("Retrieving subscriptions...", level="debug")
                if not ctx.identity:
                    ctx.log("Identity is not initialized", level="error")
                    return None
                
                ctx.log(f"Identity tenant_id: {ctx.identity.tenant_id}", level="debug")
                subscriptions = ctx.identity.get_subscriptions()
                ctx.log(f"Found {len(subscriptions)} subscriptions", level="debug")
                
                result = [{"name": sub.display_name, "id": sub.subscription_id, "state": sub.state} for sub in subscriptions]
                ctx.output(result, "Available Azure Subscriptions")
                return result
            except Exception as e:
                ctx.log(f"Error listing subscriptions: {str(e)}", level="error")
                import traceback
                ctx.log(traceback.format_exc(), level="debug")
                return None
        return func
            
    # Handle subscription get command
    elif group_name == "subscription" and cmd_name == "get":
        @pass_context
        def func(ctx, subscription_id):
            """Get details of a specific subscription."""
            try:
                ctx.log(f"Getting subscription with ID: {subscription_id}", level="debug")
                subscription = ctx.identity.get_subscription(subscription_id)
                result = {
                    "id": subscription.subscription_id,
                    "name": subscription.subscription.display_name,
                    "state": subscription.subscription.state
                }
                ctx.output(result, f"Subscription: {subscription.subscription.display_name}")
                return result
            except Exception as e:
                ctx.log(f"Error getting subscription details: {str(e)}", level="error")
                import traceback
                ctx.log(traceback.format_exc(), level="debug")
                return None
        return func
    
    # Handle resource-group list command
    elif group_name == "resource-group" and cmd_name == "list":
        @pass_context
        def func(ctx, subscription_id=None):
            """List resource groups in a subscription."""
            try:
                ctx.log("Retrieving resource groups...", level="debug")
                if not ctx.identity:
                    ctx.log("Identity is not initialized", level="error")
                    return None
                
                # Use the provided subscription ID or the default one
                subscription = None
                if subscription_id:
                    ctx.log(f"Using provided subscription ID: {subscription_id}", level="debug")
                    subscription = ctx.identity.get_subscription(subscription_id)
                elif ctx.subscription:
                    ctx.log(f"Using default subscription ID: {ctx.subscription.subscription_id}", level="debug")
                    subscription = ctx.subscription
                else:
                    ctx.log("No subscription ID provided and no default subscription set", level="error")
                    return None
                
                # Get resource groups using the resource_client
                resource_groups = list(subscription.resource_client.resource_groups.list())
                ctx.log(f"Found {len(resource_groups)} resource groups", level="debug")
                
                # Format the result
                result = [
                    {
                        "name": rg.name,
                        "location": rg.location,
                        "provisioning_state": rg.properties.provisioning_state if hasattr(rg, 'properties') and hasattr(rg.properties, 'provisioning_state') else "Unknown"
                    }
                    for rg in resource_groups
                ]
                
                ctx.output(result, f"Resource Groups in Subscription: {subscription.subscription_id}")
                return result
            except Exception as e:
                ctx.log(f"Error listing resource groups: {str(e)}", level="error")
                import traceback
                ctx.log(traceback.format_exc(), level="debug")
                return None
        return func
    
    # Handle knowledge-pipeline command
    elif group_name == "knowledge-pipeline" and cmd_name == "run":
        @pass_context
        def func(ctx, source_path, temp_path=None, format_json=None):
            """Run the knowledge pipeline to process documents."""
            try:
                ctx.log(f"Starting knowledge pipeline processing from {source_path}", level="debug")
                
                if not ctx.subscription:
                    ctx.log("Subscription is not initialized", level="error")
                    return None
                
                # Set default paths if not provided
                if not temp_path:
                    temp_path = os.path.join(os.getcwd(), "temp_json")
                    os.makedirs(temp_path, exist_ok=True)
                
                if not format_json:
                    # Try to find format.json in standard locations
                    potential_paths = [
                        os.path.join(os.getcwd(), "format.json"),
                        os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_pipeline_from_docx", "format.json")
                    ]
                    for path in potential_paths:
                        if os.path.exists(path):
                            format_json = path
                            break
                    
                    if not format_json:
                        ctx.log("format.json not found. Please specify the path using --format-json", level="error")
                        return None
                
                # Load processed files tracking
                processed_files = _load_processed_files()
                
                # Process the source path (file or directory)
                if os.path.isfile(source_path):
                    ctx.log(f"Processing single file: {source_path}")
                    if source_path.lower().endswith('.docx'):
                        _process_document(ctx, source_path, temp_path, format_json, processed_files)
                elif os.path.isdir(source_path):
                    ctx.log(f"Processing directory: {source_path}")
                    _process_directory(ctx, source_path, temp_path, format_json, processed_files)
                else:
                    ctx.log(f"Source path {source_path} does not exist", level="error")
                    return None
                
                return {"status": "completed", "message": "Document processing completed"}
            except Exception as e:
                ctx.log(f"Error in knowledge pipeline: {str(e)}", level="error")
                import traceback
                ctx.log(traceback.format_exc(), level="debug")
                return None
        return func
        
    # Handle pipeline run command
    elif group_name == "pipeline" and cmd_name == "run":
        @pass_context
        def func(ctx, manage_indexes=False, recreate_indexes=False, config=None):
            """Run the knowledge pipeline to process documents from Azure Blob Storage."""
            try:
                ctx.log("Starting knowledge pipeline...", level="info")
                
                if not ctx.subscription:
                    ctx.log("Subscription is not initialized", level="error")
                    return None
                
                # Load configuration
                pipeline_config = {}
                
                # Load from provided config file or .env
                if config and os.path.exists(config):
                    load_dotenv(config)
                else:
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
                    ctx.log(f"Missing required configuration: {', '.join(missing)}", level="error")
                    ctx.log("Set these variables in your .env file or provide a config file with --config", level="error")
                    return None
                
                # Call the pipeline function
                ctx.log("Starting run_pipeline function with the following parameters:", level="debug")
                ctx.log(f"- manage_indexes: {manage_indexes}", level="debug")
                ctx.log(f"- recreate_indexes: {recreate_indexes}", level="debug")
                
                result = run_pipeline(pipeline_config, manage_indexes, recreate_indexes)
                ctx.log("Pipeline execution completed", level="success")
                return {"status": "completed"}
            except Exception as e:
                ctx.log(f"Error in knowledge pipeline: {str(e)}", level="error")
                import traceback
                ctx.log(traceback.format_exc(), level="debug")
                return None
        return func
    
    # Handle pipeline create-indexes command
    elif group_name == "pipeline" and cmd_name == "create-indexes":
        @pass_context
        def func(ctx, core_index="knowledge-core", detailed_index="knowledge-detailed", recreate=False):
            """Create or update search indexes for the knowledge pipeline."""
            try:
                ctx.log("Starting index creation for knowledge pipeline...", level="info")
                
                if not ctx.subscription:
                    ctx.log("Subscription is not initialized", level="error")
                    return None
                
                # Load environment variables
                load_dotenv()
                
                search_service_name = os.getenv("SEARCH_SERVICE_NAME")
                resource_group_name = os.getenv("RESOURCE_GROUP")
                
                if not search_service_name:
                    ctx.log("SEARCH_SERVICE_NAME environment variable not set", level="error")
                    return None
                
                if not resource_group_name:
                    ctx.log("RESOURCE_GROUP environment variable not set", level="error")
                    return None
                
                # Get the search service
                ctx.log(f"Getting search service: {search_service_name}", level="debug")
                resource_group = ctx.subscription.get_resource_group(resource_group_name)
                search_service = ctx.subscription.get_search_service(search_service_name)
                
                if not search_service:
                    ctx.log(f"Search service {search_service_name} not found", level="error")
                    return None
                
                # Get the search index client
                search_index_client = search_service.get_index_client()
                
                # Override index names if provided
                if core_index:
                    os.environ["CORE_INDEX_NAME"] = core_index
                if detailed_index:
                    os.environ["DETAILED_INDEX_NAME"] = detailed_index
                
                # Import required functions dynamically to avoid circular imports
                try:
                    from .knowledge_pipeline import manage_azure_search_indexes
                    manage_azure_search_indexes(search_index_client, core_index, detailed_index, recreate)
                    ctx.log("Index creation completed successfully", level="success")
                    return {"status": "completed", "indexes": [core_index, detailed_index]}
                except ImportError:
                    ctx.log("knowledge_pipeline module not found. Index management functionality not available.", level="error")
                    return None
                
            except Exception as e:
                ctx.log(f"Error creating indexes: {str(e)}", level="error")
                import traceback
                ctx.log(traceback.format_exc(), level="debug")
                return None
        return func
    
    # Default handler for all other commands from main.py
    # Additional command handlers from main.py would be added here...
    
    # Default handler for all other commands
    else:
        # Default fallback function for any unimplemented command
        @pass_context
        def func(ctx, **kwargs):
            """Auto-generated handler for the command."""
            # Show which command was tried
            cmd_path = f"{group_name} {cmd_name}"
            
            ctx.log(f"Command {cmd_path} not fully implemented yet", level="error")
            
            # Show a helpful message about available commands
            ctx.log("Run --help to see available commands", level="info")
            return None
            
        # Set the function name for better debugging
        func.__name__ = func_name
        return func

# Helper functions for knowledge pipeline processing
def _load_processed_files():
    """Load the list of previously processed files from checkpoint file."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        logger.warning(f"Error loading processed files checkpoint {CHECKPOINT_FILE}: {e}")
        return set()

def _save_processed_file(filename):
    """Save a successfully processed file to the checkpoint file."""
    try:
        with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
            f.write(filename + "\n")
    except Exception as e:
        logger.warning(f"Error saving processed file checkpoint {filename}: {e}")

def _log_failed_file(filename_with_error):
    """Log a failed file for retry later."""
    try:
        with open(FAILED_FILE_LOG, "a", encoding="utf-8") as f:
            f.write(filename_with_error + "\n")
    except Exception as e:
        logger.warning(f"Error logging failed file {filename_with_error}: {e}")

def _process_document(ctx, file_path, temp_path, format_json, processed_files):
    """Process a single document."""
    # Implementation would integrate with DocParsing and other components
    # from the knowledge pipeline
    pass

def _process_directory(ctx, dir_path, temp_path, format_json, processed_files):
    """Process all documents in a directory."""
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith('.docx') and file not in processed_files:
                file_path = os.path.join(root, file)
                _process_document(ctx, file_path, temp_path, format_json, processed_files)

#############################################################
# KNOWLEDGE PIPELINE FUNCTION FROM wrapper.py
#############################################################

def run_pipeline(config: Dict[str, Any], manage_indexes: bool = False, recreate_indexes: bool = False):
    """
    Main function to orchestrate the entire knowledge pipeline process.

    Args:
        config (dict): Loaded configuration dictionary.
        manage_indexes (bool): If True, run index management before processing.
        recreate_indexes (bool): If True and manage_indexes is True, delete existing indexes first.
    """
    print("🚀 Starting Knowledge Pipeline...")

    # --- Initialization ---
    try:
        from .knowledge_pipeline import initialize_clients, load_format, delete_files_in_directory
        from .knowledge_pipeline import load_processed_files, load_failed_files, get_all_files_in_directory
        from .knowledge_pipeline import DocParsing, MultiProcessHandler, manage_azure_search_indexes
    except ImportError as e:
        logger.critical(f"CRITICAL ERROR: Knowledge Pipeline modules could not be imported: {e}")
        logger.critical("Please ensure the required libraries are correctly installed.")
        return False

    clients = initialize_clients(config)
    if not clients:
        print("❌ Pipeline aborted due to client initialization failure.")
        return False

    # Load the JSON format structure needed by DocParsing
    try:
        json_format_structure = load_format(config["FORMAT_JSON_PATH"])
        if not json_format_structure:
            print(f"❌ Error loading format JSON from {config['FORMAT_JSON_PATH']}. Aborting.")
            return False
    except Exception as e:
        print(f"❌ Error loading format JSON from {config['FORMAT_JSON_PATH']}: {e}. Aborting.")
        return False

    # --- Optional: Index Management ---
    if manage_indexes:
        print("\n🛠️ Starting Index Management...")
        manage_azure_search_indexes(
            index_client=clients["search_index_client"],
            core_index_name=config["CORE_INDEX_NAME"],
            detailed_index_name=config["DETAILED_INDEX_NAME"],
            recreate=recreate_indexes
        )
        print("🛠️ Index Management Complete.")

    # --- File Discovery and Filtering ---
    print("\n🔍 Discovering files in Azure Blob Storage...")
    processed_files = load_processed_files()

    files_to_process = []
    try:
        # Use the container client obtained during initialization
        blob_container_client = clients["container_client"]
        blob_list = blob_container_client.list_blobs() # List blobs

        for blob in blob_list:
            if blob.name.lower().endswith(".docx") and blob.name not in processed_files:
                folder_path = os.path.dirname(blob.name)
                file_name = os.path.basename(blob.name)
                # Store blob path for direct access later
                files_to_process.append({'folder': folder_path, 'name': file_name, 'blob_path': blob.name})
            elif blob.name in processed_files:
                logger.info(f"   Skipping already processed file: {blob.name}")

        print(f"   Found {len(files_to_process)} new DOCX files to process.")

    except Exception as e:
        print(f"❌ Error listing blobs in container '{config['CONTAINER_NAME']}': {e}")
        logger.exception("Blob listing failed.")
        return False

    # --- Processing Loop ---
    if not files_to_process:
        print("\n✅ No new files to process. Pipeline finished.")
        return True

    print(f"\n⚙️ Processing {len(files_to_process)} documents...")
    all_extracted_json_paths = [] # Collect paths of JSONs generated by DocParsing

    from io import BytesIO
    from docx import Document

    for file_info in files_to_process:
        folder = file_info['folder']
        file_name = file_info['name']
        blob_path = file_info['blob_path'] # Use the direct blob path
        doc_name_no_ext = os.path.splitext(file_name)[0]

        print(f"\n   Processing: {blob_path}")

        # Clean temp directory *before* processing each DOCX
        logger.info(f"      Cleaning temporary directory: {config['TEMP_PATH']}")
        delete_files_in_directory(config['TEMP_PATH'])

        try:
            # 1. Download DOCX content
            print(f"      Downloading {blob_path}...")
            blob_client = blob_container_client.get_blob_client(blob_path)
            blob_content = blob_client.download_blob().readall()
            byte_stream = BytesIO(blob_content)
            docx_document = Document(byte_stream)
            print("         ✔️ Downloaded and opened successfully.")

            # 2. Parse DOCX to JSON using DocParsing
            print("      Parsing document with DocParsing...")
            parser = DocParsing(
                doc_instance=docx_document,
                client=clients["azure_oai_client"],
                json_format=json_format_structure,
                domain="-", # Or derive domain from folder
                sub_domain=folder if folder else "root", # Use folder as sub-domain
                model_name=config["MODEL_NAME"],
                doc_name=doc_name_no_ext
            )
            # This method handles extraction, AI call, and saving JSONs
            parser.doc_to_json(output_dir=config['TEMP_PATH'])

            # 3. Collect generated JSON file paths for this DOCX
            generated_jsons = get_all_files_in_directory(config['TEMP_PATH'])
            if generated_jsons:
                logger.info(f"      ✔️ DocParsing generated {len(generated_jsons)} JSON file(s) in {config['TEMP_PATH']}.")
                all_extracted_json_paths.extend(generated_jsons)
                # Mark original DOCX blob_path as processed *only after successful parsing*
                _save_processed_file(blob_path) # Save the full blob path
                logger.info(f"      ✔️ Marked '{blob_path}' as processed (parsing stage).")
            else:
                logger.warning(f"      ⚠️ DocParsing did not generate any JSON files for {blob_path}. Skipping ingestion.")
                _log_failed_file(f"{blob_path} - DocParsing generated no JSONs")

        except Exception as e:
            print(f"❌ Error processing document {blob_path}: {e}")
            logger.exception(f"Failed processing {blob_path}")
            _log_failed_file(f"{blob_path} - Error: {e}")
            continue # Continue to the next file

    # --- Ingestion Phase ---
    if not all_extracted_json_paths:
        print("\nℹ️ No JSON files were generated by DocParsing. Nothing to ingest.")
    else:
        print(f"\n🚚 Starting ingestion phase for {len(all_extracted_json_paths)} generated JSON file(s)...")
        try:
            ingestion_handler = MultiProcessHandler(
                json_paths=all_extracted_json_paths,
                client_core=clients["search_client_core"],
                client_detail=clients["search_client_detail"],
                oai_client=clients["azure_oai_client"],
                embedding_model=config["MODEL_NAME"]
            )
            prepared_records = ingestion_handler.process_all_documents()
            if prepared_records:
                ingestion_handler.upload_to_azure_search(prepared_records)
            else:
                print("   ⚠️ No records were successfully prepared. Skipping Azure Search upload.")
        except Exception as e:
            print(f"❌ An error occurred during the ingestion phase: {e}")
            logger.exception("Ingestion phase failed.")
            return False

    print("\n🏁 Knowledge Pipeline finished.")
    return True

# Main entry point for the CLI
def main():
    """Main entry point for the extended azwrap CLI."""
    try:
        cli = create_cli()
        return cli(standalone_mode=False)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    main()