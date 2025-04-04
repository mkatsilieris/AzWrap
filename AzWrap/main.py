import argparse
import os
import sys
from dotenv import load_dotenv

# Import wrapper classes
from wrapper import (
    Identity, 
    Subscription,
    ResourceGroup,
    SearchService,
    AIService
)

def list_subscriptions(identity):
    """List available Azure subscriptions."""
    subscriptions = identity.get_subscriptions()
    print(f"Found {len(subscriptions)} subscriptions:")
    for sub in subscriptions:
        print(f"  - {sub.display_name} ({sub.subscription_id})")

def list_resource_groups(subscription):
    """List resource groups in a subscription."""
    resource_groups = subscription.resource_client.resource_groups.list()
    print(f"Resource groups in subscription {subscription.subscription.display_name}:")
    for group in resource_groups:
        print(f"  - {group.name} (Location: {group.location})")

def list_search_services(subscription):
    """List Azure AI Search services in a subscription."""
    services = subscription.get_search_services()
    print(f"Found {len(services)} search services:")
    for service in services:
        print(f"  - {service.name} (SKU: {service.sku.name}, Location: {service.location})")

def list_ai_services(subscription):
    """List Azure OpenAI services in a subscription."""
    cognitive_client = subscription.get_cognitive_client()
    resource_groups = subscription.resource_client.resource_groups.list()
    
    print("Azure OpenAI services:")
    found = 0
    
    for group in resource_groups:
        try:
            accounts = cognitive_client.accounts.list_by_resource_group(group.name)
            for account in accounts:
                if account.kind.lower() == "openai":
                    found += 1
                    print(f"  - {account.name} (Resource Group: {group.name}, Location: {account.location})")
        except Exception as e:
            pass
    
    if found == 0:
        print("  No OpenAI services found.")

def main():
    """Main entry point for the azwrap CLI."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Azure Wrapper (AzWrap) CLI tool")

    parser.add_argument("--auth-method", choices=["service_principal", "interactive"], default="service_principal", help="Authentication method to use (default: service_principal)")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List subscriptions command
    parser_list_subs = subparsers.add_parser("list-subscriptions", help="List available Azure subscriptions")
    
    # List resource groups command
    parser_list_rg = subparsers.add_parser("list-resource-groups", help="List resource groups in a subscription")
    parser_list_rg.add_argument("--subscription-id", "-s", help="Azure subscription ID", required=False)
    
    # List search services command
    parser_list_search = subparsers.add_parser("list-search", help="List Azure AI Search services")
    parser_list_search.add_argument("--subscription-id", "-s", help="Azure subscription ID", required=False)
    
    # List AI services command
    parser_list_ai = subparsers.add_parser("list-ai", help="List Azure OpenAI services")
    parser_list_ai.add_argument("--subscription-id", "-s", help="Azure subscription ID", required=False)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if any Azure credentials are available
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    
    if not all([tenant_id, client_id, client_secret]):
        print("ERROR: Azure credentials not found in environment variables.")
        print("Please set AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET.")
        # sys.exit(1)
    
    # Create identity object
    identity = Identity(tenant_id, client_id, client_secret, auth_method=args.auth_method)
    
    # Process commands
    if args.command == "list-subscriptions":
        list_subscriptions(identity)
    
    elif args.command == "list-resource-groups":
        # Use provided subscription ID or default
        sub_id = args.subscription_id or subscription_id
        if not sub_id:
            print("Error: Subscription ID not provided and AZURE_SUBSCRIPTION_ID not set.")
            sys.exit(1)
        
        subscription = identity.get_subscription(sub_id)
        if not subscription:
            print(f"Error: Subscription {sub_id} not found.")
            sys.exit(1)
        
        list_resource_groups(subscription)
    
    elif args.command == "list-search":
        # Use provided subscription ID or default
        sub_id = args.subscription_id or subscription_id
        if not sub_id:
            print("Error: Subscription ID not provided and AZURE_SUBSCRIPTION_ID not set.")
            sys.exit(1)
        
        subscription = identity.get_subscription(sub_id)
        if not subscription:
            print(f"Error: Subscription {sub_id} not found.")
            sys.exit(1)
        
        list_search_services(subscription)
    
    elif args.command == "list-ai":
        # Use provided subscription ID or default
        sub_id = args.subscription_id or subscription_id
        if not sub_id:
            print("Error: Subscription ID not provided and AZURE_SUBSCRIPTION_ID not set.")
            sys.exit(1)
        
        subscription = identity.get_subscription(sub_id)
        if not subscription:
            print(f"Error: Subscription {sub_id} not found.")
            sys.exit(1)
        
        list_ai_services(subscription)
    
    else:
        # If no command provided, show help
        parser.print_help()


if __name__ == "__main__":
    main()