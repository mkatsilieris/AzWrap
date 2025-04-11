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
    def __init__(self, json_paths: List[str] , client_core , client_detail , oai_client ):
        """
        Initializes the class with a list of JSON paths.
        """
        self.json_paths = json_paths
        self.client_core = client_core
        self.client_detail = client_detail
        self.oai_client = oai_client

    def process_documents(self) -> List[Dict]:
        """
        Processes multiple JSON documents and returns a list of processed records for each document.
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

        Args:
            client (AzureOpenAI): OpenAI client
            texts (List[str]): List of texts to embed
            model (str): Embedding model to use
        
        Returns:
            List[List[float]]: List of embeddings
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
        Uploads the processed records to Azure Search indexes. Before uploading, this method generates embeddings:
        - For the core record: an embedding for 'non_llm_summary' is generated and stored in 'embedding_summary'.
        - For each detailed record (step): embeddings for 'step_name' and 'step_content' are generated and stored in
            'embedding_title' and 'embedding_content', respectively.
        
        :param all_records: List of processed records (each a dict with keys 'core' and 'detailed').
        :param core_index_name: Name of the core index.
        :param detailed_index_name: Name of the detailed index.
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

