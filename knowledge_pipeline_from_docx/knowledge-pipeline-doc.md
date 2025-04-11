# Knowledge Pipeline from DOCX Documentation

This document provides an overview of the knowledge pipeline that processes DOCX files, extracts structured information using AI, and stores it in Azure Cognitive Search indices.

## Pipeline Overview

The pipeline performs the following key functions:
1. Downloads DOCX files from Azure Blob Storage
2. Parses DOCX files to extract process information
3. Uses Azure OpenAI to structure the content according to a predefined format
4. Processes the structured data to prepare it for indexing
5. Generates embeddings for search optimization
6. Uploads the processed data to Azure Cognitive Search indices

## Architecture

The system consists of four main components:

1. **Main Pipeline Coordinator** (`main.py`) - Orchestrates the entire process
2. **Document Parser** (`doc_parsing.py`) - Extracts and structures content from DOCX files
3. **JSON Processor** (`json_processing.py`) - Transforms parsed content into standardized records
4. **Multi-Process Handler** (`ingestion.py`) - Processes multiple documents and manages uploads

## Component Breakdown

### 1. Main Pipeline Coordinator

**File**: `main.py`

This is the main entry point that orchestrates the entire pipeline.

#### Class Structure:
The main file doesn't define a class but runs as a procedural script with multiple utility functions.

#### Key Functions:

- `initialize_clients()`: 
  - Sets up connections to Azure services (Storage, Search, OpenAI)
  - Returns a dictionary containing all client instances
  - Authenticates with Azure using service principal credentials

- `load_format(format_path)`: 
  - Loads JSON template file that defines document structure
  - Returns the loaded JSON format configuration

- `delete_files_in_directory(directory_path)`: 
  - Deletes all files in a given directory to clean up temporary storage
  - Logs each deleted file

- `get_all_files_in_directory(directory_path)`: 
  - Returns a list of file paths in the directory, excluding 'Metadata' files
  - Logs each found file

- `get_azure_oai(sub, rg, target_account_name)`: 
  - Gets the Azure OpenAI Client for AI operations
  - Finds the account by name and initializes the client

- `load_processed_files()`, `save_processed_file(filename)`, `log_failed_file(filename)`, `load_failed_files()`:
  - Checkpoint management functions for tracking processed and failed files

- `main()`: 
  - Orchestrates the entire pipeline execution
  - Handles file discovery, processing, and error logging
  - Tracks progress using checkpoint files

#### Processing Flow:

1. Initializes all required Azure clients
2. Loads checkpoint data to track processed and failed files
3. Identifies files that need processing
4. For each file:
   - Downloads from Azure Blob Storage
   - Parses using DocParsing
   - Processes JSON output using MultiProcessHandler
   - Uploads to Azure Cognitive Search
   - Updates checkpoint files

### 2. Document Parser (DocParsing)

**File**: `doc_parsing.py`

Handles extraction of content from DOCX files and structured conversion using AI.

#### Class Structure:

```python
class DocParsing:
    def __init__(self, doc_instance, client, json_file, domain, sub_domain, model_name, doc_name):
        # Initialize with document, Azure OpenAI client, and metadata
        self.client = client  # Azure OpenAI client
        self.doc = doc_instance  # Python-docx Document object
        self.doc_name = doc_name  # Name of the document 
        self.format = json_file  # JSON format template
        self.domain = domain  # Domain classification
        self.model_name = model_name  # OpenAI model name
        self.sub_domain = sub_domain  # Sub-domain classification
```

#### Key Methods:

- `get_section_header_lines(section)`: 
  - Extracts text lines from a section's header
  - Processes both paragraphs and tables in the header

- `parse_header_lines(header_lines)`: 
  - Analyzes header text to extract process titles
  - Filters out metadata text like edition info and page numbers

- `extract_header_info(section)`: 
  - Combines header extraction and parsing to get process titles
  - Returns the header title or None if extraction fails

- `iterate_block_items_with_section(doc)`: 
  - Iterates through document blocks (paragraphs and tables) with section tracking
  - Identifies section boundaries and yields elements with their section index

- `extract_table_data(table)`: 
  - Extracts text from tables, converting rows to formatted strings
  - Joins cell text with separators for readability

- `is_single_process(doc, doc_name)`: 
  - Analyzes document to determine if it contains single or multiple processes
  - Returns a tuple with boolean indicator and process title

- `extract_data()`: 
  - Main extraction function that processes document content
  - Handles both single-process and multi-process documents differently
  - Returns a dictionary with headers as keys and content as values

- `update_json(data_format, content, name)`: 
  - Sends document content to Azure OpenAI with structured prompts
  - Returns structured JSON from the AI model response

- `process_and_generate_json(response_str, output_file)`: 
  - Cleans AI response and formats it as proper JSON
  - Adds domain and subdomain metadata
  - Writes result to a file

- `doc_to_json(doc_name=None, output_dir="temp_json")`: 
  - Main conversion method that orchestrates the entire extraction process
  - Extracts data, processes each section, and saves results as JSON files

#### Processing Flow:

1. Analyzes document structure to determine if it contains single or multiple processes
2. Extracts content from paragraphs and tables based on sections
3. Sends content to Azure OpenAI with a structured prompt
4. Processes AI response and saves as JSON files

### 3. JSON Processor (ProcessHandler)

**File**: `json_processing.py`

Transforms the JSON output from document parsing into standardized records for indexing.

#### Class Structure:

```python
class ProcessHandler:
    def __init__(self, json_path: str):
        # Initialize with path to a JSON file
        # Loads and parses the JSON file containing process information
        self.json_data = None  # Will hold the loaded JSON data
        # Load JSON file and store its content
        with open(json_path, 'r', encoding='utf-8') as file:
            self.json_data = json.load(file)
```

#### Key Methods:

- `generate_process_id(process_name: str, short_description: str) -> int`: 
  - Creates a unique ID for a process using SHA-256 hashing
  - Combines process name and description for uniqueness
  - Returns a string representation of the hash-derived ID

- `generate_step_id(process_name: str, step_name: str, step_content: str) -> int`: 
  - Creates unique IDs for process steps
  - Combines process name, step name, and content for uniqueness
  - Returns a string representation of the hash-derived ID

- `prepare_core_df_record(process_id: int) -> Dict`: 
  - Prepares a record for the core search index
  - Creates a comprehensive non-LLM summary combining process attributes
  - Returns a dictionary with core process information

- `prepare_detailed_df_records(process_id: int) -> List[Dict]`: 
  - Prepares records for the detailed search index
  - Creates an introduction record (step 0) and records for each step
  - Returns a list of dictionaries with step information

- `prepare_for_upload() -> List[Dict]`: 
  - Coordinates the preparation of both core and detailed records
  - Generates process ID and calls preparation methods
  - Returns a tuple with core record and list of detailed records

#### Processing Flow:

1. Loads and parses a JSON file containing process information
2. Generates unique IDs for the process and its steps
3. Prepares a core record with process metadata and summary
4. Prepares detailed records for each step of the process
5. Returns both record types ready for upload

### 4. Multi-Process Handler (MultiProcessHandler)

**File**: `ingestion.py`

Handles processing multiple documents and uploading to Azure Cognitive Search.

#### Class Structure:

```python
class MultiProcessHandler:
    def __init__(self, json_paths: List[str], client_core, client_detail, oai_client):
        # Initialize with list of JSON paths and necessary clients
        self.json_paths = json_paths  # List of paths to JSON files
        self.client_core = client_core  # Azure Search client for core index
        self.client_detail = client_detail  # Azure Search client for detailed index
        self.oai_client = oai_client  # Azure OpenAI client for embeddings
```

#### Key Methods:

- `process_documents() -> List[Dict]`: 
  - Processes multiple JSON documents
  - Uses ProcessHandler to prepare records for each document
  - Returns a list containing core and detailed records for each document

- `generate_embeddings(client: AzureOpenAI, texts: List[str], model: str = 'text-embedding-3-large') -> List[List[float]]`: 
  - Creates vector embeddings for text fields using Azure OpenAI
  - Handles errors by returning empty embeddings for failed texts
  - Returns a list of embedding vectors for each input text

- `upload_to_azure_index(all_records: List[Dict], core_index_name: str, detailed_index_name: str) -> None`: 
  - Enriches records with AI-generated embeddings
  - Ensures proper ID formatting for Azure Search
  - Uploads records to core and detailed indices
  - Handles any errors during the upload process

#### Processing Flow:

1. Iterates through JSON files and uses ProcessHandler to prepare records
2. Generates embeddings for searchable text fields using Azure OpenAI
3. Uploads the enriched records to Azure Cognitive Search indices

## Data Flow

1. **Input**: DOCX files stored in Azure Blob Storage
2. **Intermediate**: 
   - Structured JSON files with process information
   - Core records with process metadata
   - Detailed records with step-by-step information
3. **Output**: 
   - Azure Cognitive Search core index with process summaries and embeddings
   - Azure Cognitive Search detailed index with step details and embeddings

## Data Models

### Core Index Record Structure:
- `process_id`: Unique identifier for the process
- `process_name`: Name of the process
- `doc_name`: Original document name
- `domain`: Domain classification
- `sub_domain`: Sub-domain classification
- `functional_area`: Functional area
- `functional_subarea`: Functional sub-area
- `process_group`: Process group
- `process_subgroup`: Process sub-group
- `reference_documents`: Related reference documents
- `related_products`: Related products
- `additional_information`: Additional process information
- `non_llm_summary`: Comprehensive process summary
- `embedding_summary`: Vector embedding of the summary for semantic search

### Detailed Index Record Structure:
- `id`: Unique identifier for the step
- `process_id`: ID of the parent process
- `step_number`: Order number of the step
- `step_name`: Name of the step
- `step_content`: Detailed step instructions
- `documents_used`: Documents referenced in the step
- `systems_used`: Systems used in the step
- `embedding_title`: Vector embedding of the step name
- `embedding_content`: Vector embedding of the step content

## Error Handling and Checkpointing

The pipeline implements robust error handling and checkpointing:

- Files are tracked in checkpoint files to avoid reprocessing
- Failed files are logged separately for retry
- Each processing step has error handling to prevent pipeline failure
- Detailed logging for troubleshooting

## Configuration

The pipeline uses environment variables for configuration:

- Azure authentication credentials (TENANT_ID, CLIENT_ID, CLIENT_SECRET, SUBSCRIPTION_ID)
- Storage account and container names (ACCOUNT_NAME, CONTAINER_NAME)
- Search service and index names (SEARCH_SERVICE_NAME, CORE_INDEX_NAME, DETAILED_INDEX_NAME)
- OpenAI model configuration (MODEL_NAME)
- File paths for temporary storage (TEMP_PATH, FORMAT_JSON_PATH)

This architecture enables scalable processing of process documentation with AI-powered extraction and indexing for advanced search capabilities.
