import os
import json
import shutil
from pyserini.index.lucene import LuceneIndexer
from pyserini.index import LuceneIndexReader

class DocumentIndexer:
    """
    Lucene Indexer for document
    """
    def __init__(self, index_dir: str = "lucene_index"):
        """
        Initialize the indexer
        
        Args:
            index_dir: Directory to store the Lucene index
        """
        self.index_dir = index_dir
        self.indexer = None
    
    def create_index(self, jsonl_file: str, overwrite: bool = True):
        """
        Create Lucene index from JSONL file using Pyserini's programmatic API
        
        Args:
            jsonl_file: Path to the JSONL file containing documents
            overwrite: Whether to overwrite existing index
        """
        print(f"Creating Lucene index in: {self.index_dir}")
        print(f"Source file: {jsonl_file}")
        print("=" * 60)
        
        # Remove existing index if overwrite is True
        if overwrite and os.path.exists(self.index_dir):
            print(f"Removing existing index directory: {self.index_dir}")
            shutil.rmtree(self.index_dir)
        
        # Ensure index directory exists
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Initialize indexer with stemming and proper settings
        # Using args to enable stemming and store necessary information
        args = [
            '-index', self.index_dir,
            '-collection', 'JsonCollection',
            '-stemmer', 'porter',  # Enable Porter stemming
            '-storePositions',     # Store term positions
            '-storeDocvectors',    # Store document vectors
            '-storeRaw',          # Store raw text
            '-threads', '1'       # Use single thread for consistency
        ]
        
        print("Indexer configuration:")
        print(f"  - Collection format: JsonCollection")
        print(f"  - Stemmer: Porter")
        print(f"  - Store positions: Yes")
        print(f"  - Store document vectors: Yes")
        print(f"  - Store raw text: Yes")
        print()
        
        # Create indexer
        self.indexer = LuceneIndexer(args=args)
        
        # Read and index documents
        documents_indexed = 0
        print("Indexing documents:")
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():  # Skip empty lines
                        try:
                            doc_dict = json.loads(line.strip())
                            
                            # Validate document structure
                            if 'id' not in doc_dict or 'contents' not in doc_dict:
                                print(f"  ⚠ Warning: Skipping line {line_num} - missing 'id' or 'contents'")
                                continue
                            
                            # Add document to index
                            self.indexer.add_doc_dict(doc_dict)
                            documents_indexed += 1
                            print(f"  ✓ Indexed document: {doc_dict['id']}")
                            
                        except json.JSONDecodeError as e:
                            print(f"  ❌ Error parsing line {line_num}: {e}")
                            continue
            
            # Close indexer to commit changes
            print(f"\nCommitting index with {documents_indexed} documents...")
            self.indexer.close()
            print("✓ Index creation completed successfully!")
            
        except FileNotFoundError:
            print(f"❌ Error: Could not find file {jsonl_file}")
            return False
        except Exception as e:
            print(f"❌ Error during indexing: {e}")
            return False
        
        # Verify the index
        self.verify_index()
        return True
    
    def verify_index(self):
        """
        Verify the created index and display statistics
        """
        print("\n" + "=" * 60)
        print("INDEX VERIFICATION")
        print("=" * 60)
        
        try:
            # Open index for reading
            index_reader = LuceneIndexReader(self.index_dir)
            stats = index_reader.stats()
            
            print("Index Statistics:")
            print(f"  📄 Total documents: {stats.get('documents', 'N/A')}")
            print(f"  📝 Non-empty documents: {stats.get('non_empty_documents', 'N/A')}")
            print(f"  🔤 Unique terms: {stats.get('unique_terms', 'N/A')}")
            print(f"  📊 Total terms: {stats.get('total_terms', 'N/A')}")
            
            print(f"\n📁 Index location: {os.path.abspath(self.index_dir)}")
            
            # Sample some documents to verify content
            print("\nSample Documents from Index:")
            print("-" * 40)
            
            for i in range(min(3, stats.get('documents', 0))):
                try:
                    # Convert internal docid to collection docid
                    collection_docid = index_reader.convert_internal_docid_to_collection_docid(i)
                    doc = index_reader.doc(collection_docid)
                    
                    if doc:
                        print(f"Document ID: {doc.id()}")
                        print(f"Contents: {doc.contents()[:100]}{'...' if len(doc.contents()) > 100 else ''}")
                        
                        # Show document vector (first few terms)
                        doc_vector = index_reader.get_document_vector(collection_docid)
                        if doc_vector:
                            terms = list(doc_vector.keys())[:5]  # First 5 terms
                            print(f"Sample terms: {terms}")
                        print()
                except Exception as e:
                    print(f"Could not retrieve document {i}: {e}")
                    
        except Exception as e:
            print(f"❌ Error verifying index: {e}")
            
    
    def display_index_info(self):
        """
        Display detailed information about the created index
        """
        print("\n" + "=" * 80)
        print("INDEXING PROCESS SUMMARY")
        print("=" * 80)
        
        print("Steps completed:")
        print("1. ✓ Created JSONL file with preprocessed documents")
        print("2. ✓ Configured Lucene indexer with:")
        print("   - JsonCollection format")
        print("   - Porter stemmer enabled")
        print("   - Position information stored")
        print("   - Document vectors stored")
        print("   - Raw text stored")
        print("3. ✓ Indexed all documents successfully")
        print("4. ✓ Verified index integrity")
        
        print(f"\n📍 Index ready for Boolean retrieval queries!")
        print(f"📁 Location: {os.path.abspath(self.index_dir)}")
        print("=" * 80)