
# 📄 Knowledge Pipeline Execution Guide

This guide explains how to run `main.py`, which processes and uploads structured document knowledge to Azure services. All configuration is handled via a `.env` file.

---

## 🔧 1. Setup `.env`

Create a `.env` file in the project root with the following content (replace placeholders with your actual values):

```env
# Azure Authentication
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_SUBSCRIPTION_ID=your-subscription-id

# Azure Resources
RESOURCE_GROUP=your-resource-group
TARGET_ACCOUNT_NAME=your-openai-account
ACCOUNT_NAME=your-storage-account
CONTAINER_NAME=your-container-name

# Azure Cognitive Search
SEARCH_SERVICE_NAME=your-search-service-name
CORE_INDEX_NAME=your-core-index-name
DETAILED_INDEX_NAME=your-detailed-index-name

# Model
MODEL_NAME=gpt-35-turbo  # or your deployed OpenAI model name
```

---

## 📁 2. Local Paths (used by `main.py`)

These are currently hardcoded in the script. Make sure the paths exist or modify the script if you need to change them:

```python
TEMP_PATH = r"C:\Users\agkithko\OneDrive - Netcompany\Desktop\AzWrap-1\temp_json"
FORMAT_JSON_PATH = r"C:\Users\agkithko\OneDrive - Netcompany\Desktop\AzWrap-1\knowledge_pipeline_from_docx\format.json"
```

---

## 📦 3. Install Dependencies

If a `requirements.txt` is available, run:

```bash
pip install -r requirements.txt
```

If not, make sure the following are installed (at a minimum):

```bash
pip install python-dotenv azure-identity azure-storage-blob azure-search-documents
```

---

## ▶️ 4. Run the Script

Once your `.env` is configured and paths are correct, run:

```bash
python main.py
```

The script will:
- Load all necessary configuration from `.env`
- Parse and format DOCX content using `format.json`
- Save intermediate data to the local `TEMP_PATH`
- Upload structured data to Azure Blob Storage
- Index documents into Azure Cognitive Search using two index types

---

## ✅ Done

After successful execution, check your:
- Azure Blob Storage for uploaded data
- Azure Cognitive Search for indexed content
- Local temp folder for intermediate `.json` files

---

## 🛠 Troubleshooting

- If `.env` values are not being read, confirm that `load_dotenv()` is called at the top of `main.py`
- Ensure you have the correct access rights and roles for Azure resources
- Check that the local paths exist and are accessible

---
