import os
from azure.core.exceptions import AzureError
from typing import List, Dict
import sys
from dotenv import load_dotenv
from openai import AzureOpenAI
import json

# Dynamically set the path based on this script's actual location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AZWRAP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # parent of my_tests

# Add AzWrap project root to sys.path
sys.path.append(AZWRAP_ROOT)
from knowledge_pipeline_from_docx.json_processing import ProcessHandler

load_dotenv()

class MultiProcessHandler:
    def __init__(self, json_paths: List[str], client_core, client_detail, oai_client):
        """
        Initializes the class with a list of JSON paths and necessary clients.
        
        Sets up the handler to process multiple JSON files and upload them to Azure Search
        using the provided clients.
        
        Parameters:
            json_paths: List of paths to JSON files to process
            client_core: Azure Search client for the core index
            client_detail: Azure Search client for the detailed index
            oai_client: Azure OpenAI client for generating embeddings
        """
        self.json_paths = json_paths
        self.client_core = client_core
        self.client_detail = client_detail
        self.oai_client = oai_client

    def process_documents(self) -> List[Dict]:
        """
        Processes multiple JSON documents and returns a list of processed records for each document.
        
        Iterates through each JSON file path, verifies its existence, and uses the ProcessHandler
        to prepare core and detailed records for upload.
        
        Returns:
            List of dictionaries, each containing 'core' and 'detailed' records for a document
            
        Note:
            If a file doesn't exist or an error occurs during processing, the error is logged
            and the function continues with the next file.
        """
        all_records = []

        for json_path in self.json_paths:
            if not os.path.exists(json_path):
                print(f"Error: The file {json_path} does not exist.")
                continue
            try:
                document_processor = ProcessHandler(json_path)
                core_record, detailed_records = document_processor.prepare_for_upload()
                all_records.append({
                    'core': core_record,
                    'detailed': detailed_records
                })
            except Exception as e:
                print(f"Error processing {json_path}: {e}")

        return all_records
    
    def generate_embeddings(self, client: AzureOpenAI, texts: List[str], model: str = 'text-embedding-3-large') -> List[List[float]]:
        """
        Generate embeddings for given texts.
        
        Creates vector embeddings for each text string using the Azure OpenAI embeddings API.
        Returns empty lists for any texts that fail to process or are empty.
        
        Parameters:
            client: Azure OpenAI client instance
            texts: List of text strings to generate embeddings for
            model: Name of the embedding model to use (default: 'text-embedding-3-large')
        
        Returns:
            List of embedding vectors (each a list of floats) corresponding to the input texts
        """
        embeddings = []
        for text in texts:
            if text:
                try:
                    embedding = client.embeddings.create(input=text, model=model).data[0].embedding
                    embeddings.append(embedding)
                except Exception as e:
                    embeddings.append([])
                    print("error")
            else:
                embeddings.append([])
        return embeddings

    def upload_to_azure_index(self, all_records: List[Dict], core_index_name: str, detailed_index_name: str) -> None:
        """
        Uploads the processed records to Azure Search indexes.
        
        Generates embeddings for text fields and uploads the enriched records to Azure Search.
        For core records, creates embeddings for the summary.
        For detailed records, creates embeddings for both step names and step content.
        
        Parameters:
            all_records: List of processed records (each containing 'core' and 'detailed' keys)
            core_index_name: Name of the Azure Search index for core records
            detailed_index_name: Name of the Azure Search index for detailed records
            
        Note:
            This method ensures all IDs are converted to strings before upload
            and handles any errors that occur during the upload process.
            
        Side effects:
            Records are uploaded to Azure Search if successful
            Error messages are printed to console if upload fails
        """
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential

        
        client_core = self.client_core
        client_detail = self.client_detail

        oai_client = self.oai_client
        
        try:
            for record in all_records:
                # For the core record, generate an embedding for 'non_llm_summary' if it exists.
                if 'non_llm_summary' in record['core']:
                    summary_text = record['core']['non_llm_summary']
                    embeddings = self.generate_embeddings(oai_client, [summary_text])
                    if embeddings and len(embeddings) > 0:
                        # Assign the embedding vector (list of numbers) directly
                        record['core']['embedding_summary'] = embeddings[0]
                
                # For each step record in the detailed part, generate embeddings for step_name and step_content.
                for step in record['detailed']:
                    # Ensure the step id is a string
                    if 'id' in step:
                        step['id'] = str(step['id'])
                    if 'step_name' in step:
                        name_embeddings = self.generate_embeddings(oai_client, [step['step_name']])
                        if name_embeddings and len(name_embeddings) > 0:
                            step['embedding_title'] = name_embeddings[0]
                    if 'step_content' in step:
                        content_embeddings = self.generate_embeddings(oai_client, [step['step_content']])
                        if content_embeddings and len(content_embeddings) > 0:
                            step['embedding_content'] = content_embeddings[0]
                
                record['core']['process_id'] = str(record['core']['process_id'])
                for i in record['detailed']:
                    i['id'] = str(i['id'])

                
                # Now upload the records to the respective Azure Search indexes.
                response_core = client_core.upload_documents(documents=[record['core']])
                response_detail = client_detail.upload_documents(documents=record['detailed'])
                print(f"Successfully uploaded records for {record['core'].get('process_name', 'Unknown')}")
        except Exception as e:
            print(f"Error uploading records: {e}")

