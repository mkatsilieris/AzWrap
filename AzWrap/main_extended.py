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
from pathlib import Path
from datetime import timedelta
from typing import Dict, Any, List, Optional, Union

from dotenv import load_dotenv
import azure.search.documents.indexes.models as azsdim


# Import from wrapper_extended if available
try:
    # Import additional classes or functions from wrapper_extended
    from .wrapper_extended import (
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