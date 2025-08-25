import os
import traceback

from raw_data import SAMPLE_DOCUMENTS
from preprocessing import DocumentPreprocessor
from indexing import DocumentIndexer
from retrieval import BooleanRetrieval

def main():
    print("PYSERINI BOOLEAN RETRIEVAL ASSIGNMENT")
    
    data_dir = "data"
    jsonl_file = os.path.join(data_dir, "documents.jsonl")
    index_dir = "lucene_index"
    
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        print("\n STEP 1: DOCUMENT PREPROCESSING")
        
        preprocessor = DocumentPreprocessor()
        
        print("Original Documents:")
        for doc in SAMPLE_DOCUMENTS[:3]:
            print(f"{doc['id']} : {doc['contents']}")
        print(f" ... and {len(SAMPLE_DOCUMENTS) - 3} more documents")
        
        processed_docs = preprocessor.preprocess_documents(SAMPLE_DOCUMENTS, manual=True)
        
        preprocessor.save_to_jsonl(processed_docs, jsonl_file)
        
        preprocessor.display_preprocessing_summary(SAMPLE_DOCUMENTS, processed_docs)
        
        print("\n STEP 2: LUCENE INDEXING")
        
        indexer = DocumentIndexer(index_dir)
        success = indexer.create_index(jsonl_file, overwrite=True)
        
        if not success:
            print("Indexing Failed")
            return
        
        indexer.display_index_info()
        
        print("\n STEP 3: Boolean Retrieval")
        
        retrieval = BooleanRetrieval(index_dir)
        
        # INI BUAT TEST QUERIES AJA
        retrieval.run_test_queries()
        
        print("\n" + "=" * 80)
        print("INTERACTIVE BOOLEAN QUERY MODE")
        print("=" * 80)
        print("Enter Boolean queries (or 'quit' to exit):")
        print("Examples: 'dog AND cat', 'search OR retrieval', 'cat AND NOT dog'")
        print("-" * 80)
        
        while True:
            try:
                user_query = input("\nEnter query: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_query:
                    continue
                
                print(f"\n🔍 Searching for: {user_query}")
                
                # Perform search
                results = retrieval.search_boolean(user_query)
                
                # Display results
                print(f"📄 Results: {results if results else 'No matches found'}")
                
                if results:
                    print(f"📊 Total matches: {len(results)}")
                    
                
            except KeyboardInterrupt:
                print("\n\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

        
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        
if __name__ == "__main__":
    main()