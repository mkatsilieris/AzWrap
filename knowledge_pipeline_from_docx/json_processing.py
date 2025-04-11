import hashlib
import json
from typing import List, Dict

class ProcessHandler:
    def __init__(self, json_path: str):
        """
        Initializes the class with the path to a JSON file.
        
        Loads and parses the JSON file containing process information.
        Prints information about the loading process and the process name.
        
        Parameters:
            json_path: String path to the JSON file to load
            
        Raises:
            Exception: If the JSON file cannot be loaded or parsed
        """
        print(f"🗂️ Loading JSON file: {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                self.json_data = json.load(file)
            print("✔️ JSON file loaded successfully")
            print(f"📋 Process Name: {self.json_data.get('process_name', 'N/A')}")
        except Exception as e:
            print(f"❌ Error loading JSON file: {e}")
            raise

    def generate_process_id(self, process_name: str, short_description: str) -> int:
        """
        Generate a unique integer ID for the process based on its name and short description.
        
        Creates a SHA-256 hash of the combined process name and description,
        then converts it to an integer ID.
        
        Parameters:
            process_name: Name of the process
            short_description: Brief description of the process
            
        Returns:
            String representation of the process ID derived from the hash
        """
        print(f"🔢 Generating Process ID")
        print(f"   🏷️ Process Name: {process_name}")
        print(f"   📝 Short Description: {short_description}")
        
        content_to_hash = f"{process_name}-{short_description}"
        hashed_content = hashlib.sha256(content_to_hash.encode('utf-8')).hexdigest()
        
        # Convert the hex string to an integer and return only the first 10 digits of the integer
        full_id = int(hashed_content, 16)
        process_id = str(full_id)
        
        print(f"✅ Generated Process ID: {process_id}")
        return process_id

    def generate_step_id(self, process_name: str, step_name: str, step_content: str) -> int:
        """
        Generate a unique integer ID for the step.
        
        Creates a SHA-256 hash of the combined process name, step name, and step content,
        then converts it to an integer ID.
        
        Parameters:
            process_name: Name of the parent process
            step_name: Name of the step
            step_content: Content/description of the step
            
        Returns:
            String representation of the step ID derived from the hash
        """
        print(f"🔢 Generating Step ID")
        print(f"   🏷️ Process Name: {process_name}")
        print(f"   📝 Step Name: {step_name}")
        
        content_to_hash = f"{process_name}-{step_name}-{step_content}"
        hashed_content = hashlib.sha256(content_to_hash.encode('utf-8')).hexdigest()
        
        # Convert the hex string to an integer and return only the first 10 digits of the integer
        full_id = int(hashed_content, 16)
        step_id = str(full_id)
        
        print(f"✅ Generated Step ID: {step_id}")
        return step_id

    def prepare_core_df_record(self, process_id: int) -> Dict:
        """
        Prepare record for core_df index.
        
        Creates a dictionary containing the main process information
        and a non-LLM summary combining various process attributes.
        
        Parameters:
            process_id: The unique ID for this process
            
        Returns:
            Dictionary containing the core process information formatted for database storage
        """
        print("📊 Preparing Core DataFrame Record")
        
        # Prepare steps information
        steps_info = []
        for step in self.json_data.get('steps', []):
            step_text = f"Βήμα {step['step_number']} {step['step_name']}"
            steps_info.append(step_text)

        # Prepare summary
        summary_parts = [
            "Εισαγωγή:", self.json_data.get('introduction', ''),
            "Σύντομη περιγραφή:", self.json_data.get('short_description', ''),
            "Αναλυτικά βήματα:", "\n".join(steps_info),
            "Οικογένεια προιόντων:", ", ".join(self.json_data.get('related_products', [])),
            "Έγγραφα αναφοράς:", ", ".join(self.json_data.get('reference_documents', []))
        ]
        non_llm_summary = "\n\n".join(summary_parts)

        # Prepare core record
        core_record = {
            'process_id': process_id,
            'process_name': self.json_data.get('process_name', ''),
            'doc_name': self.json_data.get('doc_name', '').split('.')[0],
            'domain': self.json_data.get('domain', ''),
            'sub_domain': self.json_data.get('subdomain', ''),
            'functional_area': self.json_data.get('functional_area', ''),
            'functional_subarea': self.json_data.get('functional_subarea', ''),
            'process_group': self.json_data.get('process_group', ''),
            'process_subgroup': self.json_data.get('process_subgroup', ''),
            'reference_documents': ', '.join(self.json_data.get('reference_documents', [])),
            'related_products': ', '.join(self.json_data.get('related_products', [])),
            'additional_information': self.json_data.get('additional_information', ''),
            'non_llm_summary': non_llm_summary.strip()
        }

        print("✅ Core DataFrame Record prepared successfully")
        return core_record

    def prepare_detailed_df_records(self, process_id: int) -> List[Dict]:
        """
        Prepare records for detailed_df index.
        
        Creates a list of dictionaries, each containing information about a step
        in the process, including an introduction record (step 0) and all regular steps.
        
        Parameters:
            process_id: The unique ID for the parent process
            
        Returns:
            List of dictionaries containing detailed step information formatted for database storage
        """
        print("📑 Preparing Detailed DataFrame Records")
        detailed_records = []

        # Generate Process ID
        process_name = self.json_data.get('process_name', '')
        short_description = self.json_data.get('short_description', '')
        process_id = self.generate_process_id(process_name, short_description)

        # Add Introduction (step 0)
        intro_content = (
            f"Εισαγωγή:\n{self.json_data.get('introduction', '')}\n\n"
            f"Σύντομη περιγραφή:\n{self.json_data.get('short_description', '')}\n\n"
            f"Οικογένεια προιόντων:\n{', '.join(self.json_data.get('related_products', []))}\n\n"
            f"Έγγραφα αναφοράς:\n{', '.join(self.json_data.get('reference_documents', []))}"
        )
        
        intro_record = {
            'id': self.generate_step_id(process_name, "Εισαγωγή", intro_content),
            'process_id': process_id,
            'step_number': 0,
            'step_name': "Εισαγωγή",
            'step_content': intro_content.strip(),
            'documents_used': None,
            'systems_used': None
        }
        detailed_records.append(intro_record)

        # Add regular steps
        print(f"📝 Total Steps: {len(self.json_data.get('steps', []))}")
        for step in self.json_data.get('steps', []):
            step_content = step.get('step_description', '')
            record = {
                'id': self.generate_step_id(process_name, step['step_name'], step_content),
                'process_id': process_id,
                'step_number': int(step['step_number']),
                'step_name': step['step_name'],
                'step_content': step_content,
                'documents_used': ', '.join(step.get('documents_used', [])),
                'systems_used': ', '.join(step.get('systems_used', []))
            }
            detailed_records.append(record)
            print(f"   ✔️ Step {record['step_number']}: {record['step_name']}")

        print("✅ Detailed DataFrame Records prepared successfully")
        return detailed_records

    def prepare_for_upload(self) -> List[Dict]:
        """
        Prepare all records for upload from the JSON data.
        
        Coordinates the generation of process IDs and the preparation
        of both core and detailed records for database upload.
        
        Returns:
            Tuple containing the core record dictionary and a list of detailed record dictionaries
        """
        print("🚀 Preparing records for upload")
        
        # Prepare core record
        process_name = self.json_data.get('process_name', '')
        short_description = self.json_data.get('short_description', '')
        process_id = self.generate_process_id(process_name, short_description)
        core_record = self.prepare_core_df_record(process_id)

        # Prepare detailed records
        detailed_records = self.prepare_detailed_df_records(process_id)

        print("🏁 Upload preparation completed")
        # Combine the core record with the detailed records
        return core_record, detailed_records

# Example usage function with enhanced logging
def process_json_for_upload(json_path: str) -> List[Dict]:
    '''
    Process a JSON file containing process data and prepare it for database upload.
    
    Creates a ProcessHandler instance to handle the JSON file,
    then prepares both core and detailed records for upload.
    
    Parameters:
        json_path: String path to the JSON file to process
        
    Returns:
        Tuple containing the core record dictionary and a list of detailed record dictionaries
        
    Raises:
        Exception: If there's an error processing the JSON file
    '''
    print(f"🔍 Processing JSON file: {json_path}")
    try:
        document_processor = ProcessHandler(json_path)
        core, detail = document_processor.prepare_for_upload()
        
        print("\n📊 Core Record Summary:")
        print(f"   🏷️ Process Name: {core.get('process_name', 'N/A')}")
        print(f"   🗂️ Domain: {core.get('domain', 'N/A')}")
        print(f"   📝 Sub-domain: {core.get('sub_domain', 'N/A')}")
        
        print(f"\n📑 Detailed Records:")
        print(f"   📝 Total Records: {len(detail)}")
        
        return core, detail
    except Exception as e:
        print(f"❌ Error processing JSON file: {e}")
        raise

# Example usage with a path to the JSON file:
if __name__ == "__main__":
    json_file_path = r"C:\Users\agkithko\OneDrive - Netcompany\Desktop\AzWrap-1\temp_json\J514_001_2023-ΕΚΤΑΜΙΕΥΣΗ ΚΑΤΑΝΑΛΩΤΙΚΟΥ ΔΑΝΕΙΟΥ_WF_single_process.json"
    core, detail = process_json_for_upload(json_file_path)

    for i in core:
        print(i , core[i])
    for y in detail:
        print(y)