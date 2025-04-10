import os
import sys
from typing import Dict, Any

# === Dynamic Path Setup ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AZWRAP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(AZWRAP_ROOT)

# === Imports from AzWrap ===
from AzWrap.wrapper import (
    Identity, Subscription, ResourceGroup, 
    SearchService, SearchIndex,get_std_vector_search
)

import azure.search.documents.indexes.models as azsdim

import os
import logging
from typing import Dict, Any
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)
from dotenv import load_dotenv


# === Config ===
load_dotenv()
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

print(SEARCH_SERVICE_NAME,
TARGET_ACCOUNT_NAME,
RESOURCE_GROUP ,
TENANT_ID ,
CLIENT_ID ,
CLIENT_SECRET ,
SUBSCRIPTION_ID,
CORE_INDEX_NAME ,
DETAILED_INDEX_NAME ,
ACCOUNT_NAME)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_vector_search_configuration() -> VectorSearch:
    """
    Create vector search configuration with exhaustive KNN algorithm.
    
    Returns:
        VectorSearch: Configured vector search settings
    """
    return VectorSearch(
        algorithms=[
            ExhaustiveKnnAlgorithmConfiguration(
                name='vector-config',
                parameters=ExhaustiveKnnParameters(
                    metric="cosine"
                )
            )
        ],
        profiles=[
            VectorSearchProfile(
                name='vector-search-profile', 
                algorithm_configuration_name='vector-config'
            )
        ]
    )

def create_enhanced_core_df_index(vector_config: VectorSearch) -> SearchIndex:
    """
    Create index schema for core document dataframe.
    
    Args:
        vector_config (VectorSearch): Vector search configuration
    
    Returns:
        SearchIndex: Configured search index for core documents
    """
    fields=[
        SimpleField(
            name="process_id", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            filterable=True, 
            retrievable=True, 
            key=True, 
            normalizer_name='lowercase'
        ),
        SearchableField(
            name="process_name", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),
        SearchableField(
            name="doc_name", 
            type=SearchFieldDataType.String,
            searchable=True, 
            retrievable=True, 
            filterable=True,
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),
        SearchableField(
            name="domain", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            filterable=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),
        SearchableField(
            name="sub_domain", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            filterable=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),
        SearchableField(
            name="functional_area", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            analyzer_name='el.lucene',
            normalizer_name='lowercase'
        ),
        SearchableField(
            name="functional_subarea", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),
        SearchableField(
            name="process_group", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),
        SearchableField(
            name="process_subgroup", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),
        SearchableField(
            name="reference_documents", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),
        SearchableField(
            name="related_products", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ), 
        SearchableField(
            name="additional_information", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),  
        SearchableField(
            name="non_llm_summary", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),
        SearchField(
            name="embedding_summary",
            type="Collection(Edm.Single)",
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="vector-search-profile"
        ),
    ]
    # Create semantic configuration
    semantic_config = SemanticConfiguration(
        name="enhanced-core-df-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="process_name"),
            content_fields=[
                SemanticField(field_name="non_llm_summary"),
            ],
            keywords_fields=[
                SemanticField(field_name="domain"),
                SemanticField(field_name="sub_domain"),
                SemanticField(field_name="functional_area"),
                SemanticField(field_name="functional_subarea"),
                SemanticField(field_name="process_group"),
                SemanticField(field_name="process_subgroup")
            ]
        )
    )
    # Create and return the index
    return SearchIndex(
        name=CORE_INDEX_NAME,
        fields=fields,
        semantic_search=SemanticSearch(configurations=[semantic_config]),
        vector_search=vector_config
    )

def create_enhanced_detailed_df_index(vector_config: VectorSearch) -> SearchIndex:
    """
    Create index schema for detailed document dataframe.
    
    Args:
        vector_config (VectorSearch): Vector search configuration
    
    Returns:
        SearchIndex: Configured search index for detailed documents
    """
    fields=[
        SimpleField(
            name="id", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            filterable=True, 
            retrievable=True, 
            key=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),
        SimpleField(
            name="process_id", 
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True, 
            retrievable=True
        ),
        SimpleField(
            name="step_number", 
            type=SearchFieldDataType.Int64,
            searchable=True,
            sortable=True,
            filterable=True, 
            retrievable=True
        ),
        SearchableField(
            name="step_name", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),
        SearchableField(
            name="step_content", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),
        SearchableField(
            name="documents_used", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),
        SearchableField(
            name="systems_used", 
            type=SearchFieldDataType.String, 
            searchable=True, 
            retrievable=True, 
            analyzer_name='el.lucene', 
            normalizer_name='lowercase'
        ),
        SearchField(
            name="embedding_title",
            type="Collection(Edm.Single)",
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="vector-search-profile"
        ),
        SearchField(
            name="embedding_content",
            type="Collection(Edm.Single)",
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="vector-search-profile"
        )
    ]
    semantic_config = SemanticConfiguration(
        name="enhanced-detailed-df-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="step_name"),
            content_fields=[SemanticField(field_name="step_content")],
            keywords_fields=[]
        )
    )
    # Create and return the index
    return SearchIndex(
        name=DETAILED_INDEX_NAME,
        fields=fields,
        semantic_search=SemanticSearch(configurations=[semantic_config]),
        vector_search=vector_config
    )

def list_indexes(index_client: SearchIndexClient):
    """
    List all indexes in the Azure AI Search service.
    
    Args:
        index_client (SearchIndexClient): The search index client
    """
    indexes = index_client.list_indexes()
    index_names = [index.name for index in indexes]
    logger.info(f"Number of indexes: {len(index_names)}")
    logger.info("Indexes:")
    for name in index_names:
        logger.info(f"- {name}")

def manage_azure_search_indexes():
    """
    Manage Azure Cognitive Search indexes.
    
    Args:
        env_file_path (str): Path to the .env file
        excel_file_path (str): Path to the Excel file with document data
    """
    try:

        identity = Identity(tenant_id=TENANT_ID, client_id=CLIENT_ID , client_secret=CLIENT_SECRET)
        subscription_info = identity.get_subscription(SUBSCRIPTION_ID)
        sub = Subscription(identity=identity, subscription=subscription_info, subscription_id=subscription_info.subscription_id)

        rg = sub.get_resource_group(RESOURCE_GROUP)
        storage_account = rg.get_storage_account(ACCOUNT_NAME)

        search_service = sub.get_search_service(SEARCH_SERVICE_NAME)
        index_client = search_service.get_index_client()

        # List existing indexes before deletion
        logger.info("Existing indexes before deletion:")
        list_indexes(index_client)
        
        # Create vector search configuration
        vector_config = create_vector_search_configuration()
        
        # Create index configurations
        core_df_index = create_enhanced_core_df_index(vector_config)
        detailed_df_index = create_enhanced_detailed_df_index(vector_config)
        
        # Delete existing indexes
        try:
            index_client.delete_index(core_df_index)
            logger.info(f"Deleted index '{core_df_index.name}' successfully!")
        except Exception as e:
            logger.warning(f"Error deleting core index: {e}")
        
        try:
            index_client.delete_index(detailed_df_index)
            logger.info(f"Deleted index '{detailed_df_index.name}' successfully!")
        except Exception as e:
            logger.warning(f"Error deleting detailed index: {e}")
        # List existing indexes after deletion
        logger.info("Existing indexes after deletion:")
        list_indexes(index_client)
        
        # Create new indexes
        try:
            index_client.create_or_update_index(core_df_index)
            logger.info(f"Create index '{core_df_index.name}' successfully!") 
        except Exception as e:
            logger.warning(f"Error creating core_df index: {e}")
        try:    
            index_client.create_or_update_index(detailed_df_index)
            logger.info(f"Create index '{detailed_df_index.name}' successfully!")
        except Exception as e:
            logger.warning(f"Error creating detailed index: {e}")
        
        logger.info("All indexes created successfully!")
        # List existing indexes after creation
        logger.info("Existing indexes after creation:")
        list_indexes(index_client)
    
    except Exception as e:
        logger.error(f"An error occurred during index management: {e}")

def main():
    """
    Main entry point for the Azure Search Index setup script.
    """
    
    # Run index management
    manage_azure_search_indexes() 


if __name__ == "__main__":
    main()
