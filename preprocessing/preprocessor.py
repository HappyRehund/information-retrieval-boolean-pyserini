import json
import string
from typing import List, Dict
from pyserini.analysis import get_lucene_analyzer, JAnalyzerUtils
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class DocumentPreprocessor:
    """
    Text preprocessing class for documents before indexing
    """
    
    def __init__(self):
        """
        Initialize preprocessor
        """
        self.stemmer = PorterStemmer()
        self.analyzer = get_lucene_analyzer(stemming=True, stopwords=True)
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Download NLTK stopwords")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase"""
        return text.lower()
    
    def remove_stopwords(self, words: List[str]) -> List[str]:
        """Remove stopwords from list of words"""
        return [word for word in words if word.lower() not in self.stop_words]
    
    def stem_words(self, words: List[str]) -> List[str]:
        """Apply Porter stemming to words"""
        return [self.stemmer.stem(word) for word in words]
    
    def clean_whitespace(self, text: str) -> str:
        """Remove extra whitespace"""
        return ' '.join(text.split())
    
    def preprocess_manual(self, text: str) -> str:
        """
        Manual preprocessing pipeline
        """
        print(f"Original text: {text}")
        
        # Step 1: Convert to lowercase
        text = self.to_lowercase(text)
        print(f"After lowercase: {text}")
        
        # Step 2: Remove punctuation
        text = self.remove_punctuation(text)
        print(f"After removing punctuation: {text}")
        
        # Step 3: Clean whitespace
        text = self.clean_whitespace(text)
        print(f"After cleaning whitespace: {text}")
        
        # Step 4: Tokenize, remove stopwords, and stem
        words = text.split()
        print(f"After tokenization: {words}")
        
        words = self.remove_stopwords(words)
        print(f"After removing stopwords: {words}")
        
        words = self.stem_words(words)
        print(f"After stemming: {words}")
        
        processed_text = ' '.join(words)
        print(f"Final processed text: {processed_text}")
        print("-" * 50)
        
        return processed_text
    
    def preprocess_with_analyzer(self, text: str) -> str:
        """
        Use Pyserini's analyzer for preprocessing (this will be used during indexing)
        """
        # Pyserini's analyzer will handle stemming and stopword removal during indexing
        tokens = JAnalyzerUtils.analyze(self.analyzer, text)
        processed_tokens = [str(token) for token in tokens.toArray()]
        return ' '.join(processed_tokens)
    
    def preprocess_documents(self, documents: List[Dict[str, str]], manual: bool = True) -> List[Dict[str, str]]:
        """
        Preprocess a list of documents
        
        Args:
            documents: List of document dictionaries with 'id' and 'contents'
            manual: If True, use manual preprocessing, otherwise use Pyserini analyzer
            
        Returns:
            List of preprocessed documents
        """
        processed_docs = []
        
        print("Starting document preprocessing...")
        print("=" * 60)
        
        for doc in documents:
            doc_id = doc['id']
            original_content = doc['contents']
            
            print(f"Processing document {doc_id}:")
            
            if manual:
                processed_content = self.preprocess_manual(original_content)
            else:
                processed_content = self.preprocess_with_analyzer(original_content)
                print(f"Processed with analyzer: {processed_content}")
                print("-" * 50)
            
            processed_docs.append({
                'id': doc_id,
                'contents': processed_content,
                'raw': original_content  # Keep original for reference
            })
        
        print("Preprocessing completed!")
        print("=" * 60)
        return processed_docs
    
    def save_to_jsonl(self, documents: List[Dict[str, str]], output_path: str):
        """
        Save documents to JSONL format
        """
        print(f"Saving {len(documents)} documents to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                # For JSONL, we only need id and contents
                jsonl_doc = {'id': doc['id'], 'contents': doc['contents']}
                f.write(json.dumps(jsonl_doc) + '\n')
        
        print(f"Documents saved successfully to {output_path}")
    
    def display_preprocessing_summary(self, original_docs: List[Dict], processed_docs: List[Dict]):
        """
        Display a summary of the preprocessing results
        """
        print("\n" + "="*80)
        print("PREPROCESSING SUMMARY")
        print("="*80)
        
        for _, (orig, proc) in enumerate(zip(original_docs, processed_docs)):
            print(f"\nDocument {orig['id']}:")
            print(f"  Original:  {orig['contents']}")
            print(f"  Processed: {proc['contents']}")
            
            # Show word count changes
            orig_words = len(orig['contents'].split())
            proc_words = len(proc['contents'].split())
            print(f"  Word count: {orig_words} → {proc_words} (reduced by {orig_words - proc_words})")
        
        print("\nPreprocessing steps applied:")
        print("1. ✓ Convert to lowercase")
        print("2. ✓ Remove punctuation")
        print("3. ✓ Remove extra whitespace")
        print("4. ✓ Remove stopwords")
        print("5. ✓ Apply Porter stemming")
        print("="*80)