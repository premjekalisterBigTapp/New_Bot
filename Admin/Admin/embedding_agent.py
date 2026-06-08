import os
import sys

# Add the project root to the Python path before other imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import argparse
import re
import json
import weaviate
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from weaviate.classes.query import Filter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Path setup will be done in main() when needed

# Direct Azure OpenAI clients (no CrewAI wrappers)
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Hardcoded Azure OpenAI configuration (per request)
# Chat/completion endpoint
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# Separate embedding endpoint (different resource)
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_OPENAI_EMBEDDING_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Chat client (uses main endpoint)
_azure_chat_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT.rstrip("/"),
)

# Embedding client (uses separate embedding endpoint)
_azure_embed_client = AzureOpenAI(
    api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
    api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT.rstrip("/"),
)

def azure_chat(system_text: str, user_text: str) -> str:
    resp = _azure_chat_client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

def azure_embed(text: str) -> list:
    emb = _azure_embed_client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        input=text,
    )
    return emb.data[0].embedding

# Define project root and source_db path for robust execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DB_PATH = os.path.join(PROJECT_ROOT, "Admin", "source_db")
DEBUG_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "Admin", "debug_chunks")


# Setup logging
log_directory = "Admin/logs"
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, "embedding_agent.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_all_products():
    """Gets all unique product names by scanning filenames in the source_db directory."""
    products = set()
    if not os.path.isdir(SOURCE_DB_PATH):
        logger.error(f"Source directory not found: {SOURCE_DB_PATH}")
        return []
        
    for doc_type_folder in os.listdir(SOURCE_DB_PATH):
        type_dir_path = os.path.join(SOURCE_DB_PATH, doc_type_folder)
        if os.path.isdir(type_dir_path):
            # Special handling for PDFs where product is a sub-folder
            if doc_type_folder.lower() == 'pdfs':
                for product_folder in os.listdir(type_dir_path):
                    product_path = os.path.join(type_dir_path, product_folder)
                    if os.path.isdir(product_path):
                        products.add(product_folder)
            else:
                # For other types, extract product from filename like "Travel_benefits.txt"
                for filename in os.listdir(type_dir_path):
                    if '_' in filename:
                        product_name = filename.split('_')[0]
                        products.add(product_name)
    logger.info(f"Discovered products: {list(products)}")
    return list(products)

def chunk_benefits(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """Chunks benefits text with compulsory character-level overlap.

    - Enforces exact chunk_size and chunk_overlap windows (step = chunk_size - chunk_overlap).
    - Guarantees that every chunk (except the first) starts with the last `chunk_overlap`
      characters of the previous chunk, regardless of paragraph boundaries.
    """
    logger.info(f"Chunking benefits file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Guardrails
    if chunk_overlap >= chunk_size:
        logger.warning("chunk_overlap >= chunk_size; adjusting overlap to chunk_size // 10")
        chunk_overlap = max(0, chunk_size // 10)

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step

    logger.info(f"Created {len(chunks)} chunks from {os.path.basename(file_path)} (size={chunk_size}, overlap={chunk_overlap})")
    return chunks

def chunk_faqs(file_path: str):
    """Chunks a FAQ file where each Q&A pair is a chunk."""
    logger.info(f"Chunking FAQ file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Split by "Q:" but keep the delimiter
    qa_pairs = re.split(r'(?=^Q:)', text, flags=re.MULTILINE)
    chunks = [pair.strip() for pair in qa_pairs if pair.strip()]
    logger.info(f"Created {len(chunks)} chunks from {os.path.basename(file_path)}")
    return chunks

def chunk_policy_md(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Chunks a markdown file using a sliding window approach to keep related sections together.
    """
    logger.info(f"Chunking policy file with sliding window: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # Sensible separators for markdown
    )
    
    chunks = text_splitter.split_text(text)
    
    logger.info(f"Created {len(chunks)} chunks from {os.path.basename(file_path)}")
    return chunks

def generate_hypothetical_questions(chunk: str):
    """Generates 20 comprehensive, unique, exclusion-aware questions for a given text chunk using an LLM."""
    logger.info(f"Generating questions for chunk starting with: '{chunk[:80]}...'")
    
    system_text = (
        "You are an expert at generating comprehensive hypothetical questions from insurance policy text chunks.\n"
        "Generate EXACTLY 20 detailed, UNIQUE questions anchored to the provided chunk.\n"
        "Non-overlapping: no two questions should be answerable by the same single clause or exclusion; avoid paraphrases and near-duplicates.\n"
        "Each question must combine related aspects (benefits, scenarios, conditions, limits, documentation, timing windows, exclusions) rather than focusing on a single point.\n"
        "Exclusion-aware: distinguish between policy-wide GENERAL EXCLUSIONS and SECTION-SPECIFIC EXCLUSIONS present in the chunk; craft questions that test both and clarify their interaction or precedence when relevant.\n"
        "Incorporate edge cases commonly found in travel insurance policies when present in the chunk: vaccination requirements and exemptions, age-based applicability, pre-existing conditions and optional covers, publicly known events and travel alerts, quarantine and COVID-specific provisions, police/carrier report requirements, pair-and-sets clause, itemized bill requirements, automatic extensions, payment-before-cover warranty, sanctions/subrogation, and definitions that change applicability (Common Carrier, Trip, Immediate Family Member).\n"
        "Do not include tier-specific questions; keep questions generally applicable across plans.\n"
        "Output strictly as JSON with a single key 'questions' mapped to an array of 20 strings. No extra text.\n"
        "Examples of multi-aspect questions (illustrative, adapt to the chunk):\n"
        "- How do overseas and Singapore follow-up medical expense windows, documentation requirements, and section-specific exclusions interact when treatment starts overseas but continues at home?\n"
        "- Under travel cancellation, postponement, and curtailment, what constitutes covered reasons versus general or section exclusions (e.g., publicly known events, travel alerts), and how do refunds from carriers affect claims?\n"
        "- For baggage loss and personal documents, how do police report timing, carrier Property Irregularity Reports, pair-and-sets clause, and unattended property rules combine to determine claim eligibility?\n"
    )
    user_text = f"Here is the text chunk:\n\n---\n{chunk}\n---"

    try:
        txt = azure_chat(system_text, user_text)
        json_str = txt.replace("```json", "").replace("```", "").strip()
        questions_obj = json.loads(json_str)
        questions = questions_obj.get("questions", [])
        questions = questions[:20]
        if not isinstance(questions, list) or len(questions) == 0:
            logger.warning("LLM returned empty or invalid question list.")
            return []
        logger.info(f"Successfully generated {len(questions)} comprehensive questions.")
        return questions
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM response: {e}\nResponse: {txt}")
        return []
    except Exception as e:
        logger.error(f"An error occurred during question generation: {e}")
        return []


def save_chunks_to_debug_folder(product_name, all_objects):
    """
    Save chunks and metadata to debug folder for analysis.
    """
    # Create debug folder structure
    product_debug_path = os.path.join(DEBUG_OUTPUT_PATH, product_name)
    os.makedirs(product_debug_path, exist_ok=True)
    
    logger.info(f"Saving {len(all_objects)} chunks to debug folder: {product_debug_path}")
    
    # Save detailed chunk analysis
    chunk_analysis = {
        "product_name": product_name,
        "total_chunks": len(all_objects),
        "chunks_by_doc_type": {},
        "chunks_by_source": {},
        "empty_chunks": [],
        "valid_chunks": [],
        "chunk_details": []
    }
    
    for i, obj in enumerate(all_objects):
        chunk_id = f"{product_name}_{obj['doc_type']}_{i+1}"
        content = obj.get('content', '')
        
        # Analyze chunk
        is_empty = not content or not content.strip()
        content_length = len(content) if content else 0
        
        chunk_detail = {
            "chunk_id": chunk_id,
            "chunk_index": i + 1,
            "doc_type": obj.get('doc_type', 'unknown'),
            "source_file": obj.get('source_file', 'unknown'),
            "content_length": content_length,
            "is_empty": is_empty,
            "has_questions": bool(obj.get('questions', [])),
            "questions_count": len(obj.get('questions', [])),
            "content_preview": content
        }
        
        chunk_analysis["chunk_details"].append(chunk_detail)
        
        # Track by doc_type
        doc_type = obj.get('doc_type', 'unknown')
        if doc_type not in chunk_analysis["chunks_by_doc_type"]:
            chunk_analysis["chunks_by_doc_type"][doc_type] = {"total": 0, "empty": 0, "valid": 0}
        chunk_analysis["chunks_by_doc_type"][doc_type]["total"] += 1
        
        # Track by source file
        source_file = obj.get('source_file', 'unknown')
        if source_file not in chunk_analysis["chunks_by_source"]:
            chunk_analysis["chunks_by_source"][source_file] = {"total": 0, "empty": 0, "valid": 0}
        chunk_analysis["chunks_by_source"][source_file]["total"] += 1
        
        if is_empty:
            chunk_analysis["empty_chunks"].append(chunk_id)
            chunk_analysis["chunks_by_doc_type"][doc_type]["empty"] += 1
            chunk_analysis["chunks_by_source"][source_file]["empty"] += 1
            logger.warning(f"Empty chunk detected: {chunk_id}")
        else:
            chunk_analysis["valid_chunks"].append(chunk_id)
            chunk_analysis["chunks_by_doc_type"][doc_type]["valid"] += 1
            chunk_analysis["chunks_by_source"][source_file]["valid"] += 1
        
        # Save individual chunk file
        chunk_filename = f"{chunk_id}.json"
        chunk_filepath = os.path.join(product_debug_path, chunk_filename)
        
        with open(chunk_filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "chunk_id": chunk_id,
                "metadata": {
                    "product_name": obj.get('product_name', ''),
                    "doc_type": obj.get('doc_type', ''),
                    "source_file": obj.get('source_file', ''),
                    "chunk_index": i + 1,
                    "content_length": content_length,
                    "is_empty": is_empty
                },
                "content": content,
                "questions": obj.get('questions', [])
            }, f, indent=2, ensure_ascii=False)
    
    # Save summary analysis
    summary_filepath = os.path.join(product_debug_path, f"{product_name}_chunk_analysis.json")
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(chunk_analysis, f, indent=2, ensure_ascii=False)
    
    # Save human-readable summary
    summary_txt_filepath = os.path.join(product_debug_path, f"{product_name}_summary.txt")
    with open(summary_txt_filepath, 'w', encoding='utf-8') as f:
        f.write(f"CHUNK ANALYSIS SUMMARY FOR {product_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Chunks: {chunk_analysis['total_chunks']}\n")
        f.write(f"Empty Chunks: {len(chunk_analysis['empty_chunks'])}\n")
        f.write(f"Valid Chunks: {len(chunk_analysis['valid_chunks'])}\n\n")
        
        f.write("BREAKDOWN BY DOCUMENT TYPE:\n")
        f.write("-" * 30 + "\n")
        for doc_type, stats in chunk_analysis["chunks_by_doc_type"].items():
            f.write(f"{doc_type}: {stats['total']} total, {stats['valid']} valid, {stats['empty']} empty\n")
        
        f.write("\nBREAKDOWN BY SOURCE FILE:\n")
        f.write("-" * 30 + "\n")
        for source_file, stats in chunk_analysis["chunks_by_source"].items():
            f.write(f"{source_file}: {stats['total']} total, {stats['valid']} valid, {stats['empty']} empty\n")
        
        if chunk_analysis['empty_chunks']:
            f.write("\nEMPTY CHUNKS:\n")
            f.write("-" * 15 + "\n")
            for chunk_id in chunk_analysis['empty_chunks']:
                f.write(f"- {chunk_id}\n")
    
    logger.info(f"Debug analysis saved to: {summary_filepath}")
    logger.info(f"Found {len(chunk_analysis['empty_chunks'])} empty chunks out of {chunk_analysis['total_chunks']} total chunks")
    
    return chunk_analysis


def embed_product(product_name, weaviate_client, auto_proceed=False):
    """
    Processes documents for a single product and embeds them into the 'Insurance_Knowledge_Base' collection using raw Weaviate.
    """
    logger.info(f"Starting embedding process for product: {product_name}")
    
    collection_name = "Insurance_Knowledge_Base"
    
    try:
        docs_collection = weaviate_client.collections.get(collection_name)
    except Exception as e:
        logger.error(f"Failed to get Weaviate collection '{collection_name}': {e}")
        return

    # Preflight: check for existing product data and optionally delete
    try:
        existing_count = docs_collection.aggregate.over_all(
            filters=Filter.by_property("product_name").equal(product_name)
        ).total_count
        if existing_count > 0:
            print(f"\nFound {existing_count} existing objects for product '{product_name}' in collection '{collection_name}'.")
            while True:
                try:
                    resp = input("Delete them before continuing? (yes/no): ").strip().lower()
                except KeyboardInterrupt:
                    print("\nProcess interrupted.")
                    return
                if resp in ("y", "yes"):
                    try:
                        docs_collection.data.delete_many(
                            where=Filter.by_property("product_name").equal(product_name)
                        )
                        logger.info("Deleted %d existing objects for product: %s", existing_count, product_name)
                    except Exception as de:
                        logger.error("Failed to delete existing objects for %s: %s", product_name, de)
                        return
                    break
                elif resp in ("n", "no"):
                    logger.info("Keeping existing objects for product: %s; new data will be added.", product_name)
                    break
                else:
                    print("Please enter 'yes' or 'no'.")
    except Exception as e:
        logger.warning("Preflight check for existing product data failed (%s): %s", product_name, e)

    # Define file paths and their corresponding chunking functions and doc_types
    files_to_process = [
        {"path": os.path.join(SOURCE_DB_PATH, "benefits", f"{product_name}_benefits.txt"), "chunker": chunk_benefits, "doc_type": "benefits"},
        {"path": os.path.join(SOURCE_DB_PATH, "FAQ", f"{product_name}_FAQs.txt"), "chunker": chunk_faqs, "doc_type": "faq"},
        {"path": os.path.join(SOURCE_DB_PATH, "policy", f"{product_name}_policy.md"), "chunker": chunk_policy_md, "doc_type": "policy"},
    ]

    all_objects = []
    benefits_chunks = []  # Store benefits chunks for display
    
    # First pass: chunk all documents
    for file_info in files_to_process:
        file_path = file_info["path"]
        if os.path.exists(file_path):
            logger.info(f"Processing file: {file_path}")
            chunks = file_info["chunker"](file_path)
            logger.info(f"Generated {len(chunks)} chunks from {os.path.basename(file_path)}")
            
            valid_chunks_count = 0
            empty_chunks_count = 0
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{product_name}_{file_info['doc_type']}_{i+1}"
                
                if not chunk or not chunk.strip():
                    logger.warning(f"Skipping empty chunk {chunk_id} from file: {os.path.basename(file_path)}")
                    empty_chunks_count += 1
                    continue

                valid_chunks_count += 1
                logger.debug(f"Processing valid chunk {chunk_id}: {len(chunk)} characters")

                # Store benefits chunks for user review
                if file_info["doc_type"] == "benefits":
                    benefits_chunks.append({
                        "chunk_id": chunk_id,
                        "content": chunk,
                        "length": len(chunk),
                        "source_file": os.path.basename(file_path)
                    })
                
                all_objects.append({
                    "content": chunk,
                    "questions": [],  # Will be generated later
                    "product_name": product_name,
                    "doc_type": file_info["doc_type"],
                    "source_file": os.path.basename(file_path)
                })
            
            logger.info(f"File {os.path.basename(file_path)} processing complete: {valid_chunks_count} valid chunks, {empty_chunks_count} empty chunks skipped")
        else:
            logger.warning(f"File not found, skipping: {file_path}")

    if not all_objects:
        logger.warning(f"No chunks generated for product: {product_name}")
        return

    # Display benefits chunks for user review
    if benefits_chunks:
        print(f"\n{'='*80}")
        print(f"BENEFITS CHUNKS FOR PRODUCT: {product_name.upper()}")
        print(f"{'='*80}")

        for i, chunk_info in enumerate(benefits_chunks, 1):
            print(f"\n--- CHUNK {i} ---")
            print(f"ID: {chunk_info['chunk_id']}")
            print(f"Length: {chunk_info['length']} characters")
            print(f"Source: {chunk_info['source_file']}")
            print(f"Content:\n{chunk_info['content']}")
            print(f"{'-'*80}")

        print(f"\nTotal Benefits Chunks: {len(benefits_chunks)}")

    # Interactive prompt (skip if auto_proceed is True)
    if not auto_proceed:
        while True:
            try:
                response = input("Proceed to question generation and embedding? (yes/no): ").strip().lower()
                if response in ['yes', 'y']:
                    logger.info("User approved proceeding to question generation and embedding")
                    break
                elif response in ['no', 'n']:
                    logger.info("User declined proceeding. Stopping process.")
                    print("Process stopped by user.")
                    return
                else:
                    print("Please enter 'yes' or 'no'.")
            except KeyboardInterrupt:
                logger.info("Process interrupted by user")
                print("\nProcess interrupted.")
                return
            except Exception as e:
                logger.error(f"Error getting user input: {e}")
                return
    else:
        logger.info("Auto-proceed mode enabled. Proceeding to question generation and embedding.")

    # Generate hypothetical questions for each chunk with progress
    total_objs = len(all_objects)
    logger.info("STEP 1/3: Generating hypothetical questions for %d chunks", total_objs)
    t0 = time.time()
    for idx, obj in enumerate(all_objects, start=1):
        try:
            chunk = obj.get('content', '')
            if chunk:
                # Progress/ETA
                elapsed = time.time() - t0
                avg = elapsed / max(1, idx - 1) if idx > 1 else 0.0
                remaining_s = max(0.0, (total_objs - idx) * avg)
                pct = (idx / total_objs) * 100.0
                logger.info(
                    "QGen %d/%d (%.1f%%) | doc_type=%s | len=%d | elapsed=%.1fs | eta=%.1fs",
                    idx, total_objs, pct, obj.get('doc_type'), len(chunk), elapsed, remaining_s,
                )
                questions = generate_hypothetical_questions(chunk)
                obj['questions'] = questions
            else:
                obj['questions'] = []
        except Exception as e:
            logger.error("Failed to generate questions for chunk %d: %s", idx, e)
            obj['questions'] = []
    total_q = sum(len(o.get('questions') or []) for o in all_objects)
    logger.info("STEP 1/3 DONE: Generated %d questions across %d chunks (%.1fs)", total_q, total_objs, time.time() - t0)

    # Save chunks to debug folder for analysis
    logger.info(f"Generated {len(all_objects)} data objects for product: {product_name}")
    logger.info("STEP 2/3: Saving debug analysis to disk")
    chunk_analysis = save_chunks_to_debug_folder(product_name, all_objects)
    logger.info("STEP 2/3 DONE: Debug analysis saved")
    
    # Log analysis summary
    logger.info(f"Chunk Analysis Summary - Total: {chunk_analysis['total_chunks']}, Valid: {len(chunk_analysis['valid_chunks'])}, Empty: {len(chunk_analysis['empty_chunks'])}")
    
    # Only proceed with Weaviate ingestion if we have valid chunks
    valid_objects = [obj for obj in all_objects if obj.get('content') and obj.get('content').strip()]
    
    if not valid_objects:
        logger.error(f"No valid chunks found for product: {product_name}. Skipping Weaviate ingestion.")
        return
    
    if len(valid_objects) != len(all_objects):
        logger.warning(f"Filtering out {len(all_objects) - len(valid_objects)} empty chunks before Weaviate ingestion")
    
    logger.info("STEP 3/3: Ingesting %d valid chunks into Weaviate...", len(valid_objects))

    try:
        logger.info("Generating embeddings and ingesting %d objects into collection '%s'...", len(valid_objects), collection_name)
        
        with docs_collection.batch.dynamic() as batch:
            ingest_start = time.time()
            total_ingest = len(valid_objects)
            for i, item in enumerate(valid_objects, start=1):
                try:
                    # Generate SEPARATE embeddings for content and questions
                    content_text = item.get('content', '')
                    questions_text = ' '.join(item.get('questions', []))
                    
                    if content_text:
                        # Generate separate embeddings (3072-dim each) using the azure_embed function
                        logger.debug(f"Generating content embedding for chunk {i+1}")
                        content_embedding = azure_embed(content_text)
                        
                        # Prepare vectors dictionary
                        vectors = {"content_vector": content_embedding}
                        
                        # Generate questions embedding if questions exist
                        if questions_text and questions_text.strip():
                            logger.debug(f"Generating questions embedding for chunk {i+1}")
                            questions_embedding = azure_embed(questions_text)
                            vectors["questions_vector"] = questions_embedding
                        
                        # Store with named vectors
                        batch.add_object(
                            properties=item,  # Same metadata structure
                            vector=vectors    # Multiple named vectors
                        )
                        
                        # Progress log every 10 items or on last item
                        if (i % 10 == 0) or (i == total_ingest):
                            pct = (i / total_ingest) * 100.0
                            elapsed = time.time() - ingest_start
                            avg = elapsed / max(1, i)
                            eta = max(0.0, (total_ingest - i) * avg)
                            logger.info("Ingest %d/%d (%.1f%%) | elapsed=%.1fs | eta=%.1fs", i, total_ingest, pct, elapsed, eta)
                    else:
                        logger.warning(f"Skipping object {i+1} - no content to embed")
                        
                except Exception as embed_error:
                    logger.error(f"Failed to generate embedding for object {i+1}: {embed_error}")
                    # Add without embedding as fallback
                    batch.add_object(properties=item)
        
        if docs_collection.batch.failed_objects:
             logger.error(f"Failed to ingest {len(docs_collection.batch.failed_objects)} objects for {product_name}.")
             for failed_obj in docs_collection.batch.failed_objects:
                 logger.error(f"Failed object: {failed_obj}")
        
        logger.info(f"Successfully ingested {len(valid_objects)} valid objects for {product_name} into Weaviate collection: {collection_name}")

    except Exception as e:
        logger.error(f"Failed to ingest documents for {product_name}: {e}")
        
    # Log final summary
    logger.info(f"Embedding process completed for {product_name}:")
    logger.info(f"  - Total chunks processed: {chunk_analysis['total_chunks']}")
    logger.info(f"  - Valid chunks ingested: {len(valid_objects)}")
    logger.info(f"  - Empty chunks filtered out: {len(chunk_analysis['empty_chunks'])}")
    logger.info(f"  - Debug files saved to: {os.path.join(DEBUG_OUTPUT_PATH, product_name)}")

def main():
    """
    Main function to parse arguments and trigger the embedding process.
    """
    parser = argparse.ArgumentParser(description="Embedding agent for processing product documents.")
    parser.add_argument("--product", type=str, help="The name of the product to process. If not provided, all products will be processed.")
    parser.add_argument("--auto-proceed", action="store_true", help="Automatically proceed without user confirmation (non-interactive mode).")
    args = parser.parse_args()

    # Create debug output directory
    os.makedirs(DEBUG_OUTPUT_PATH, exist_ok=True)
    logger.info(f"Debug output will be saved to: {DEBUG_OUTPUT_PATH}")

    # Import centralized Weaviate client from the Agentic-Bot infrastructure
    # Add Agentic-Bot to path if not already there
    agentic_bot_path = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
    if agentic_bot_path not in sys.path:
        sys.path.insert(0, agentic_bot_path)
    
    from infrastructure.vector_store import get_weaviate_client, initialize_weaviate
    client = None
    try:
        # Initialize Weaviate client first
        initialize_weaviate()
        client = get_weaviate_client()
        collection_name = "Insurance_Knowledge_Base"

        # Idempotent collection creation
        if not client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' does not exist. Creating it now.")
            client.collections.create(
                name=collection_name,
                # Configure NAMED VECTORS for separate semantic spaces (MANUAL EMBEDDINGS)
                vector_config=[
                    {
                        "name": "content_vector",
                        "vectorizer": Configure.Vectorizer.none(),  # Manual embeddings
                        "vector_index_config": Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE
                        )
                    },
                    {
                        "name": "questions_vector",
                        "vectorizer": Configure.Vectorizer.none(),  # Manual embeddings
                        "vector_index_config": Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE
                        )
                    }
                ],
                properties=[
                    # KEEP EXACT SAME METADATA (for rec_retriever_agent compatibility)
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="questions", data_type=DataType.TEXT_ARRAY),
                    Property(name="product_name", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                    Property(name="source_file", data_type=DataType.TEXT),
                ],
            )
            logger.info(f"Successfully created collection: {collection_name}")
        else:
            logger.info(f"Using existing Weaviate collection: {collection_name}")

        if args.product:
            embed_product(args.product, client, auto_proceed=args.auto_proceed)
        else:
            logger.info("No specific product specified. Processing all available products.")
            products = get_all_products()
            if not products:
                logger.warning("No product directories found in source_db.")
                return
            for product in products:
                embed_product(product, client, auto_proceed=args.auto_proceed)
    
    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")
    finally:
        if client and client.is_connected():
            client.close()
            logger.info("Weaviate client closed.")

if __name__ == "__main__":
    main()
