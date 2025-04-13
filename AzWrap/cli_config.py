"""
CLI configuration file for AzWrap.
This module contains configuration for all CLI commands and options.
"""

CLI_CONFIG = {
    "name": "azwrap",
    "description": "Azure Wrapper (AzWrap) CLI tool for managing Azure resources",
    "commands": {
        # Resource Management Commands
        "subscription": {
            "description": "Manage Azure subscriptions",
            "subcommands": {
                "list": {
                    "description": "List available Azure subscriptions",
                    "options": []
                },
                "get": {
                    "description": "Get details of a specific subscription",
                    "options": [
                        {"name": "subscription-id", "short": "s", "required": True, "help": "Azure subscription ID"}
                    ]
                }
            }
        },
        
        "resource-group": {
            "description": "Manage Azure resource groups",
            "subcommands": {
                "list": {
                    "description": "List resource groups in a subscription",
                    "options": [
                        {"name": "subscription-id", "short": "s", "required": False, "help": "Azure subscription ID"}
                    ]
                },
                "create": {
                    "description": "Create a new resource group",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Resource group name"},
                        {"name": "location", "short": "l", "required": True, "help": "Azure region (e.g., westus2)"},
                        {"name": "subscription-id", "short": "s", "required": False, "help": "Azure subscription ID"}
                    ]
                },
                "get": {
                    "description": "Get details of a specific resource group",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Resource group name"},
                        {"name": "subscription-id", "short": "s", "required": False, "help": "Azure subscription ID"}
                    ]
                }
            }
        },
        
        # Search Service Commands
        "search": {
            "description": "Manage Azure AI Search services",
            "subcommands": {
                "list": {
                    "description": "List Azure AI Search services",
                    "options": [
                        {"name": "subscription-id", "short": "s", "required": False, "help": "Azure subscription ID"},
                        {"name": "resource-group", "short": "g", "required": False, "help": "Resource group name"},
                    ]
                },
                "create": {
                    "description": "Create a new search service",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "location", "short": "l", "required": True, "help": "Azure region"},
                        {"name": "sku", "required": False, "help": "SKU (free, basic, standard, standard2, standard3)", "default": "basic"}
                    ]
                },
                "get": {
                    "description": "Get details of a specific search service",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                }
            }
        },
        
        # Index Commands
        "index": {
            "description": "Manage search indexes",
            "subcommands": {
                "list": {
                    "description": "List indexes in a search service",
                    "options": [
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "create": {
                    "description": "Create a new search index",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Index name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "fields", "short": "f", "required": True, "help": "JSON string defining fields"}
                    ]
                },
                "get": {
                    "description": "Get details of a specific index",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Index name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "delete": {
                    "description": "Delete a search index",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Index name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "force", "is_flag": True, "help": "Force deletion without confirmation"}
                    ]
                },
                "copy": {
                    "description": "Copy an index (structure and/or data)",
                    "options": [
                        {"name": "source", "required": True, "help": "Source index name"},
                        {"name": "target", "required": True, "help": "Target index name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "structure-only", "is_flag": True, "help": "Copy only the index structure, not data"},
                        {"name": "fields", "help": "Comma-separated list of fields to copy"}
                    ]
                }
            }
        },
        
        # Data Source Commands
        "datasource": {
            "description": "Manage search data sources",
            "subcommands": {
                "list": {
                    "description": "List data source connections",
                    "options": [
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "create": {
                    "description": "Create a data source connection",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Data source name"},
                        {"name": "type", "short": "t", "required": True, "help": "Data source type (e.g., azureblob, azuretable, azuresql)"},
                        {"name": "connection-string", "short": "c", "required": True, "help": "Connection string"},
                        {"name": "container", "required": True, "help": "Container name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "get": {
                    "description": "Get details of a specific data source",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Data source name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "delete": {
                    "description": "Delete a data source",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Data source name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "force", "is_flag": True, "help": "Force deletion without confirmation"}
                    ]
                }
            }
        },
        
        # Indexer Commands
        "indexer": {
            "description": "Manage search indexers",
            "subcommands": {
                "list": {
                    "description": "List indexers",
                    "options": [
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "create": {
                    "description": "Create an indexer",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Indexer name"},
                        {"name": "data-source", "short": "d", "required": True, "help": "Data source name"},
                        {"name": "target-index", "short": "i", "required": True, "help": "Target index name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "skillset", "help": "Skillset name to use with this indexer"},
                        {"name": "schedule", "help": "Indexing schedule (cron expression)"}
                    ]
                },
                "run": {
                    "description": "Run an indexer",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Indexer name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "reset": {
                    "description": "Reset an indexer",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Indexer name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "status": {
                    "description": "Get indexer status",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Indexer name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "update": {
                    "description": "Update an indexer",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Indexer name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "schedule", "help": "Indexing schedule (cron expression)"},
                        {"name": "parameters", "help": "JSON string with indexing parameters"}
                    ]
                },
                "delete": {
                    "description": "Delete an indexer",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Indexer name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "force", "is_flag": True, "help": "Force deletion without confirmation"}
                    ]
                }
            }
        },
        
        # Skillset Commands
        "skillset": {
            "description": "Manage search skillsets",
            "subcommands": {
                "list": {
                    "description": "List skillsets",
                    "options": [
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "create": {
                    "description": "Create a skillset",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Skillset name"},
                        {"name": "skills", "required": True, "help": "Skills JSON or path to JSON file"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "description", "help": "Skillset description"}
                    ]
                },
                "get": {
                    "description": "Get details of a specific skillset",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Skillset name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "update": {
                    "description": "Update a skillset",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Skillset name"},
                        {"name": "skills", "required": True, "help": "Skills JSON or path to JSON file"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "description", "help": "Skillset description"}
                    ]
                },
                "delete": {
                    "description": "Delete a skillset",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Skillset name"},
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "force", "is_flag": True, "help": "Force deletion without confirmation"}
                    ]
                }
            }
        },
        
        # AI Service Commands
        "ai": {
            "description": "Manage Azure OpenAI services and deployments",
            "subcommands": {
                "list": {
                    "description": "List Azure OpenAI services",
                    "options": [
                        {"name": "subscription-id", "short": "s", "required": False, "help": "Azure subscription ID"},
                        {"name": "resource-group", "short": "g", "required": False, "help": "Resource group name"}
                    ]
                },
                "get": {
                    "description": "Get details of a specific OpenAI service",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "models": {
                    "description": "List available models for an OpenAI service",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "location", "short": "l", "help": "Azure region (e.g., westus2)"}
                    ]
                },
                "deployments": {
                    "description": "List deployments for an OpenAI service",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "deploy": {
                    "description": "Create a new model deployment",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Deployment name"},
                        {"name": "model", "short": "m", "required": True, "help": "Model name"},
                        {"name": "service", "required": True, "help": "OpenAI service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "capacity", "short": "c", "help": "Capacity in TPM", "default": "1"},
                        {"name": "sku", "help": "SKU name", "default": "Standard"},
                        {"name": "version", "short": "v", "help": "Model version"}
                    ]
                },
                "update-deployment": {
                    "description": "Update a model deployment",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Deployment name"},
                        {"name": "service", "required": True, "help": "OpenAI service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "capacity", "short": "c", "help": "New capacity in TPM"},
                        {"name": "sku", "help": "New SKU name"}
                    ]
                },
                "delete-deployment": {
                    "description": "Delete a model deployment",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Deployment name"},
                        {"name": "service", "required": True, "help": "OpenAI service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "force", "is_flag": True, "help": "Force deletion without confirmation"}
                    ]
                },
                "embedding": {
                    "description": "Generate embeddings for text",
                    "options": [
                        {"name": "text", "short": "t", "required": True, "help": "Text to embed or path to text file"},
                        {"name": "service", "required": True, "help": "OpenAI service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "model", "short": "m", "help": "Embedding model", "default": "text-embedding-3-small"},
                        {"name": "api-version", "short": "a", "required": True, "help": "API version"}
                    ]
                },
                "chat": {
                    "description": "Generate chat completion",
                    "options": [
                        {"name": "message", "short": "m", "required": True, "help": "User message"},
                        {"name": "service", "required": True, "help": "OpenAI service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "model", "required": True, "help": "Model deployment name"},
                        {"name": "api-version", "short": "a", "required": True, "help": "API version"},
                        {"name": "temperature", "help": "Temperature (0-1)", "default": "0.7"},
                        {"name": "max-tokens", "help": "Maximum tokens", "default": "800"},
                        {"name": "system", "short": "s", "help": "System message"}
                    ]
                }
            }
        },
        
        # Storage Account Commands
        "storage": {
            "description": "Manage Azure Storage accounts",
            "subcommands": {
                "list": {
                    "description": "List storage accounts",
                    "options": [
                        {"name": "subscription-id", "short": "s", "required": False, "help": "Azure subscription ID"}
                    ]
                },
                "get": {
                    "description": "Get details of a specific storage account",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Storage account name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "create": {
                    "description": "Create a new storage account",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Storage account name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "location", "short": "l", "required": True, "help": "Azure region (e.g., westus2)"},
                        {"name": "sku", "help": "SKU name", "default": "Standard_LRS"}
                    ]
                },
                "containers": {
                    "description": "List containers in a storage account",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Storage account name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "blobs": {
                    "description": "List blobs in a container",
                    "options": [
                        {"name": "account", "short": "a", "required": True, "help": "Storage account name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "container", "short": "c", "required": True, "help": "Container name"}
                    ]
                },
                "folder-structure": {
                    "description": "Get folder structure in a container",
                    "options": [
                        {"name": "account", "short": "a", "required": True, "help": "Storage account name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "container", "short": "c", "required": True, "help": "Container name"}
                    ]
                }
            }
        }
    },
    "global_options": [
        {"name": "verbose", "short": "v", "is_flag": True, "help": "Enable verbose output"},
        {"name": "quiet", "short": "q", "is_flag": True, "help": "Suppress all output except errors and results"},
        {"name": "output", "short": "o", "help": "Output format (text, json, table)", "default": "text"},
        {"name": "config", "help": "Path to configuration file"}
    ]
}

# Extended CLI configuration with additional commands for the enhanced functionality
EXTENDED_CLI_CONFIG = {
    "name": "azwrap",
    "description": "Extended Azure Wrapper (AzWrap) CLI tool with additional functionality",
    "commands": {
        # Knowledge Pipeline Commands
        "pipeline": {
            "description": "Manage document processing and knowledge extraction pipelines",
            "subcommands": {
                "run": {
                    "description": "Run the knowledge pipeline to process documents from Azure Blob Storage",
                    "options": [
                        {"name": "manage-indexes", "is_flag": True, "help": "Run index management tasks before processing files"},
                        {"name": "recreate-indexes", "is_flag": True, "help": "Delete existing indexes before creating new ones (use with caution)"},
                        {"name": "config", "short": "c", "required": False, "help": "Path to configuration file (defaults to .env)"}
                    ]
                },
                "create-indexes": {
                    "description": "Create or update search indexes for the knowledge pipeline",
                    "options": [
                        {"name": "core-index", "required": False, "help": "Name for the core index", "default": "knowledge-core"},
                        {"name": "detailed-index", "required": False, "help": "Name for the detailed index", "default": "knowledge-detailed"},
                        {"name": "recreate", "is_flag": True, "help": "Delete existing indexes before creating new ones"}
                    ]
                }
            }
        },
        
        # Vector Search Commands
        "vector-search": {
            "description": "Manage and perform vector searches",
            "subcommands": {
                "create-config": {
                    "description": "Create a vector search configuration for an index",
                    "options": [
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "index-name", "short": "i", "required": True, "help": "Index name"},
                        {"name": "metric", "short": "m", "required": False, "help": "Distance metric (cosine, euclidean, dotProduct)", "default": "cosine"},
                        {"name": "connections", "short": "c", "required": False, "help": "Number of connections per node", "default": "4"},
                        {"name": "neighbors", "short": "n", "required": False, "help": "Size of neighbors list", "default": "400"},
                        {"name": "search-size", "required": False, "help": "Size of search list", "default": "500"}
                    ]
                },
                "hybrid-search": {
                    "description": "Perform a hybrid search combining keyword and vector searching",
                    "options": [
                        {"name": "search-service", "short": "s", "required": True, "help": "Search service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "index-name", "short": "i", "required": True, "help": "Index name"},
                        {"name": "query", "short": "q", "required": True, "help": "Search query text"},
                        {"name": "vector-field", "short": "v", "required": False, "help": "Vector field name", "default": "text_vector"},
                        {"name": "semantic", "is_flag": True, "help": "Enable semantic search"},
                        {"name": "top", "short": "t", "required": False, "help": "Number of results to return", "default": "10"}
                    ]
                }
            }
        },
        
        # Deployment Commands
        "deployment": {
            "description": "Manage deployments for Azure OpenAI and AI Search services",
            "subcommands": {
                "get": {
                    "description": "Get details of a specific deployment",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Deployment name"},
                        {"name": "service", "required": True, "help": "OpenAI service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"}
                    ]
                },
                "create": {
                    "description": "Creates a new deployment",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Deployment name"},
                        {"name": "model", "short": "m", "required": True, "help": "Model name"},
                        {"name": "service", "required": True, "help": "OpenAI service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "capacity", "short": "c", "help": "Capacity in TPM", "default": "1"},
                        {"name": "sku", "help": "SKU name", "default": "Standard"},
                        {"name": "version", "short": "v", "help": "Model version"}
                    ]
                },
                "update": {
                    "description": "Updates an existing deployment",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Deployment name"},
                        {"name": "service", "required": True, "help": "OpenAI service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "capacity", "short": "c", "help": "New capacity in TPM"},
                        {"name": "sku", "help": "New SKU name"}
                    ]
                },
                "delete": {
                    "description": "Deletes a deployment",
                    "options": [
                        {"name": "name", "short": "n", "required": True, "help": "Deployment name"},
                        {"name": "service", "required": True, "help": "OpenAI service name"},
                        {"name": "resource-group", "short": "g", "required": True, "help": "Resource group name"},
                        {"name": "force", "is_flag": True, "help": "Force deletion without confirmation"}
                    ]
                }
            }
        }
    }
}