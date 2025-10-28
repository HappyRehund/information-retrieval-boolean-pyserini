import os
from typing import List, Dict, Set, Any
from pyserini.search.lucene import LuceneSearcher
from pyserini.index import LuceneIndexReader
from preprocessing import DocumentPreprocessor

class BooleanRetrieval:
    """
    Boolean retrieval system with manual inverted index
    """
    
    def __init__(self, index_dir: str = "lucene_index"):
        """
        Initialize the Boolean retrieval system
        """
        self.index_dir = index_dir
        self.searcher: LuceneSearcher | None = None
        self.index_reader: LuceneIndexReader | None = None
        self.inverted_index = {}
        self.documents = {}
        self.preprocessor = DocumentPreprocessor()
        self.initialize_searcher()
        self.build_inverted_index()
    
    def initialize_searcher(self):
        """
        Initialize the Lucene searcher and index reader
        """
        if not os.path.exists(self.index_dir):
            raise FileNotFoundError(f"Index directory not found: {self.index_dir}")
        
        try:
            self.searcher = LuceneSearcher(self.index_dir)
            self.index_reader = LuceneIndexReader(self.index_dir)
            print("✓ Boolean retrieval system initialized")
            print(f"📁 Index: {self.index_dir}")
            if self.index_reader is not None:
                print(f"📄 Documents available: {self.index_reader.stats()['documents']}")
        except Exception as e:
            raise Exception(f"Failed to initialize searcher: {e}")
    
    def build_inverted_index(self):
        """
        Build inverted index manually from the Lucene index
        """
        if self.index_reader is None:
            raise Exception("Index reader not initialized")
            
        print("Building inverted index...")
        
        total_docs = self.index_reader.stats()['documents']
        
        for internal_docid in range(total_docs):
            try:
                # Get document ID
                collection_docid = self.index_reader.convert_internal_docid_to_collection_docid(internal_docid)
                
                # Get document
                doc = self.index_reader.doc(collection_docid)
                if doc:
                    doc_content = doc.contents()
                    # Ensure doc_content is not None
                    if doc_content is None:
                        doc_content = ""
                    self.documents[collection_docid] = doc_content
                    
                    # Get document vector (terms and frequencies)
                    doc_vector = self.index_reader.get_document_vector(collection_docid)
                    if doc_vector:
                        for term in doc_vector.keys():
                            if term not in self.inverted_index:
                                self.inverted_index[term] = set()
                            self.inverted_index[term].add(collection_docid)
                else:
                    # Handle case where document is None
                    print(f"Warning: Document {collection_docid} returned None")
                    self.documents[collection_docid] = ""
                
            except Exception as e:
                print(f"Error processing document {internal_docid}: {e}")
        
        print(f"✓ Inverted index built with {len(self.inverted_index)} unique terms")
        print(f"✓ Documents loaded: {len(self.documents)}")
        print(f"✓ Inverted index built with {len(self.inverted_index)} unique terms")
    
    def search_boolean(self, query: str, max_results: int = 100) -> List[str]:
        """
        Perform Boolean search using manual inverted index
        """
        try:
            # Clean and parse query
            query = query.strip()
            if not query:
                return []
            
            # Parse Boolean query
            result_set = self._parse_boolean_query(query)
            
            # Convert set to sorted list
            results = sorted(list(result_set))[:max_results]
            return results
            
        except Exception as e:
            print(f"Error in Boolean search: {e}")
            return []
    
    def _parse_boolean_query(self, query: str) -> Set[str]:
        """
        Parse and execute Boolean query
        """
        query = query.lower().strip()
        
        # Handle different Boolean operators
        if " and not " in query:
            return self._handle_and_not(query)
        elif " and " in query:
            return self._handle_and(query)
        elif " or " in query:
            return self._handle_or(query)
        elif " not " in query:
            return self._handle_not(query)
        else:
            # Single term query
            return self._get_documents_for_term(query)
    
    def _handle_and(self, query: str) -> Set[str]:
        """Handle AND queries"""
        terms = [term.strip() for term in query.split(" and ")]
        
        if not terms:
            return set()
        
        # Start with documents containing the first term
        result_set = self._get_documents_for_term(terms[0])
        
        # Intersect with documents containing other terms
        for term in terms[1:]:
            term_docs = self._get_documents_for_term(term)
            result_set = result_set.intersection(term_docs)
        
        return result_set
    
    def _handle_or(self, query: str) -> Set[str]:
        """Handle OR queries"""
        terms = [term.strip() for term in query.split(" or ")]
        
        result_set = set()
        
        # Union all documents containing any of the terms
        for term in terms:
            term_docs = self._get_documents_for_term(term)
            result_set = result_set.union(term_docs)
        
        return result_set
    
    def _handle_not(self, query: str) -> Set[str]:
        """Handle NOT queries (term1 NOT term2)"""
        parts = query.split(" not ")
        if len(parts) != 2:
            return set()
        
        positive_term = parts[0].strip()
        negative_term = parts[1].strip()
        
        positive_docs = self._get_documents_for_term(positive_term)
        negative_docs = self._get_documents_for_term(negative_term)
        
        # Documents with positive term but not negative term
        return positive_docs - negative_docs
    
    def _handle_and_not(self, query: str) -> Set[str]:
        """Handle AND NOT queries"""
        parts = query.split(" and not ")
        if len(parts) != 2:
            return set()
        
        # Handle multiple terms before AND NOT
        left_part = parts[0].strip()
        negative_term = parts[1].strip()
        
        # Get documents matching the left part
        if " and " in left_part:
            positive_docs = self._handle_and(left_part)
        else:
            positive_docs = self._get_documents_for_term(left_part)
        
        # Get documents to exclude
        negative_docs = self._get_documents_for_term(negative_term)
        
        # Return positive documents minus negative documents
        return positive_docs - negative_docs
    
    def _get_documents_for_term(self, term: str) -> Set[str]:
        """
        Get all documents containing a specific term
        Apply same preprocessing as during indexing
        """
        term = term.strip().lower()
        
        # Apply preprocessing (tokenization, stopword removal, stemming)
        processed_term = self.preprocessor.preprocess_text(term)
        
        # processed_term is a list of tokens, we need to search for each token
        result_docs = set()
        
        for token in processed_term:
            if token in self.inverted_index:
                # UNION all documents containing any token from this term
                result_docs = result_docs.union(self.inverted_index[token])
        
        return result_docs
    
    def explain_boolean_query(self, query: str, results: List[str]) -> str:
        """
        Provide explanation for Boolean query results
        """
        explanations = []
        
        if not results:
            explanations.append("No documents match the query.")
            return "\n".join(explanations)
        
        # Analyze query components
        query_lower = query.lower()
        
        if " and not " in query_lower:
            parts = query_lower.split(" and not ")
            positive_part = parts[0].strip()
            negative_term = parts[1].strip()
            explanations.append(f"Query requires documents with '{positive_part}' but NOT '{negative_term}'")
            
        elif " and " in query_lower:
            terms = [term.strip() for term in query_lower.split(" and ")]
            explanations.append(f"Query requires ALL terms: {terms}")
            explanations.append("Documents must contain every specified term.")
            
        elif " or " in query_lower:
            terms = [term.strip() for term in query_lower.split(" or ")]
            explanations.append(f"Query requires ANY of the terms: {terms}")
            explanations.append("Documents need to contain at least one of the specified terms.")
            
        elif " not " in query_lower:
            parts = query_lower.split(" not ")
            positive_term = parts[0].strip()
            negative_term = parts[1].strip()
            explanations.append(f"Query requires documents with '{positive_term}' but NOT '{negative_term}'")
        
        
        return "\n".join(explanations)
    
    def verify_boolean_logic(self, query: str, results: List[str]) -> Dict[str, Any]:
        """
        Verify that Boolean query results are logically correct
        """
        verification: Dict[str, Any] = {
            'query': query,
            'total_results': len(results),
            'logic_correct': True,
            'issues': [],
            'document_analysis': []
        }
        
        query_lower = query.lower()
        
        for doc_id in results:
            # Safe handling of document content
            doc_content_raw = self.documents.get(doc_id, "")
            if doc_content_raw is None:
                doc_content_raw = ""
            doc_content = str(doc_content_raw).lower()
            
            if " and not " in query_lower:
                self._verify_and_not_logic(query_lower, doc_id, verification)
            elif " and " in query_lower:
                self._verify_and_logic(query_lower, doc_id, verification)
            elif " or " in query_lower:
                self._verify_or_logic(query_lower, doc_id, verification)
            elif " not " in query_lower:
                self._verify_not_logic(query_lower, doc_id, verification)
        
        return verification
    
    def _verify_and_logic(self, query: str, doc_id: str, verification: Dict[str, Any]):
        """Verify AND logic"""
        terms = [term.strip() for term in query.split(" and ")]
        
        doc_analysis = {
            'doc_id': doc_id,
            'required_terms': terms,
            'found_terms': [],
            'missing_terms': []
        }
        
        for term in terms:
            # Check if term exists in inverted index for this document
            if doc_id in self._get_documents_for_term(term):
                doc_analysis['found_terms'].append(term)
            else:
                doc_analysis['missing_terms'].append(term)
                verification['logic_correct'] = False
                verification['issues'].append(f"Document {doc_id} missing required term: {term}")
        
        verification['document_analysis'].append(doc_analysis)
    
    def _verify_or_logic(self, query: str, doc_id: str, verification: Dict[str, Any]):
        """Verify OR logic"""
        terms = [term.strip() for term in query.split(" or ")]
        
        found_any = False
        found_terms = []
        
        for term in terms:
            if doc_id in self._get_documents_for_term(term):
                found_any = True
                found_terms.append(term)
        
        if not found_any:
            verification['logic_correct'] = False
            verification['issues'].append(f"Document {doc_id} doesn't contain any required terms: {terms}")
        
        verification['document_analysis'].append({
            'doc_id': doc_id,
            'any_of_terms': terms,
            'found_terms': found_terms
        })
    
    def _verify_and_not_logic(self, query: str, doc_id: str, verification: Dict[str, Any]):
        """Verify AND NOT logic"""
        parts = query.split(" and not ")
        if len(parts) != 2:
            return
        
        positive_part = parts[0].strip()
        negative_term = parts[1].strip()
        
        # Check if document has positive terms
        if " and " in positive_part:
            positive_terms = [term.strip() for term in positive_part.split(" and ")]
            missing_positive = []
            for term in positive_terms:
                if doc_id not in self._get_documents_for_term(term):
                    missing_positive.append(term)
            if missing_positive:
                verification['logic_correct'] = False
                verification['issues'].append(f"Document {doc_id} missing required terms: {missing_positive}")
        else:
            if doc_id not in self._get_documents_for_term(positive_part):
                verification['logic_correct'] = False
                verification['issues'].append(f"Document {doc_id} missing required term: {positive_part}")
        
        # Check if document has negative term (should not have)
        if doc_id in self._get_documents_for_term(negative_term):
            verification['logic_correct'] = False
            verification['issues'].append(f"Document {doc_id} contains excluded term: {negative_term}")
    
    def _verify_not_logic(self, query: str, doc_id: str, verification: Dict[str, Any]):
        """Verify NOT logic"""
        parts = query.split(" not ")
        if len(parts) != 2:
            return
        
        positive_term = parts[0].strip()
        negative_term = parts[1].strip()
        
        # Check positive term
        if doc_id not in self._get_documents_for_term(positive_term):
            verification['logic_correct'] = False
            verification['issues'].append(f"Document {doc_id} missing required term: {positive_term}")
        
        # Check negative term (should not have)
        if doc_id in self._get_documents_for_term(negative_term):
            verification['logic_correct'] = False
            verification['issues'].append(f"Document {doc_id} contains excluded term: {negative_term}")
    
    def run_test_queries(self) -> List[Dict[str, Any]]:
        """
        Run a set of test queries and return results
        """
        test_queries = [
            "dog AND cat",
            "dog OR cat", 
            "dog AND NOT cat",
            "dog OR short",
            "rank OR night",
        ]
        
        print("=" * 80)
        print("BOOLEAN RETRIEVAL TESTING")
        print("=" * 80)
        
        query_results: List[Dict[str, Any]] = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            print("-" * 50)
            
            # Perform search
            results = self.search_boolean(query)
            
            # Verify logic
            verification = self.verify_boolean_logic(query, results)
            
            print(f"Matching documents: {results}")
            print(f"Total matches: {len(results)}")
            
            if verification['logic_correct']:
                print("✓ Logic verification: PASSED")
            else:
                print("✓ Logic verification: FAILED")
                print("⚠ Issues found:")
                for issue in verification['issues'][:5]:  # Show first 5 issues
                    print(f"   - {issue}")
                if len(verification['issues']) > 5:
                    print(f"   ... and {len(verification['issues']) - 5} more issues")
            
            # Show explanation
            explanation = self.explain_boolean_query(query, results)
            print(f"\nExplanation:")
            print(explanation)
            
            query_results.append({
                'query': query,
                'results': results,
                'verification': verification,
                'explanation': explanation
            })
        
        # Summary
        passed = sum(1 for result in query_results if result['verification']['logic_correct'])
        total = len(query_results)
        
        print(f"\nBOOLEAN QUERY TESTING SUMMARY")
        print("=" * 80)
        print(f"Total queries tested: {total}")
        print(f"Queries passed verification: {passed}")
        print(f"Success rate: {(passed/total)*100:.1f}%")
        
        return query_results
    
    def display_inverted_index(self):
        """
        Display the inverted index for debugging
        """
        print("\n🔍 INVERTED INDEX")
        print("=" * 50)
        
        for term in sorted(self.inverted_index.keys()):
            docs = sorted(list(self.inverted_index[term]))
            print(f"{term}: {docs}")