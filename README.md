# Pyserini Boolean Retrieval Assignment
1. Rayhan Firdaus Ardian
2. Daffa Harikhsan
3. Zaky Alraiz Kadarisman

This project demonstrates Boolean information retrieval using Pyserini's Lucene indexing and search capabilities.

## 📋 Assignment Overview

The assignment implements a complete Boolean retrieval system with:
1. **Document Preprocessing** - Text cleaning, stemming, stopword removal
2. **Lucene Indexing** - Creating searchable index with Pyserini
3. **Boolean Retrieval** - Performing AND, OR, NOT queries

## 🏗️ Project Structure

```
├── data/                      
│   └── documents.jsonl       # Processed JSONL file (generated)
├── preprocessing/
│   └── preprocessor.py       # Text preprocessing utilities
├── indexing/
│   └── indexer.py           # Lucene indexing functionality
├──raw_data/                 
│   └── raw_documents.py     # 15 raw data
├── retrieval/
│   └── searcher.py          # Boolean search functionality
├── main.py                  # Main execution script
├── requirements.txt         # Python dependencies
├── setup.sh                # Automated setup script
└── README.md               # This file
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.7+
- Java 11+ (required by Pyserini)


### Setup
```bash
# Install dependencies, make sure you already installed uv in your pc
uv sync

# Download NLTK data
uv run python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```
or run this command first (Windows - git bash):
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

```bash
python main.py
```

## 🚀 Running the Assignment

Execute the main script:
```bash
uv run main.py
```

or run this command first (Windows - git bash):
```bash
source .venv/scripts/activate
```

```bash
python main.py
```

The program will:
1. **Preprocess** the 15 sample documents
2. **Index** them using Lucene
3. **Run** Boolean query tests
4. **Provide** interactive query mode

## 📊 Sample Output

### Preprocessing Results
```
Document d1:
  Original:  The cat chased a small mouse into the garden.
  Processed: cat chase small mous garden
  Word count: 9 → 5 (reduced by 4)
```

### Boolean Query Results
```
🔍 Query 1: dog AND cat
📄 Matching documents: ['d4', 'd12']
📊 Total matches: 2
✓ Logic verification: PASSED

💡 Explanation:
Query requires ALL terms: ['dog', 'cat']
Documents must contain every specified term.
```

## 🔧 Technical Details

### Preprocessing Pipeline
1. Convert to lowercase
2. Remove punctuation
3. Remove stopwords (using NLTK)
4. Apply Porter stemming
5. Clean whitespace

### Indexing Configuration
- **Collection Format**: JsonCollection
- **Stemmer**: Porter stemmer
- **Storage**: Positions, document vectors, raw text
- **Threads**: Single thread for consistency

### Boolean Query Support
- **AND** - All terms must be present
- **OR** - At least one term must be present  
- **NOT** - Exclude documents with specified terms
- **Parentheses** - Group operations for complex logic

## 🎯 Learning Objectives Achieved

✅ **Preprocessing**: Demonstrated text cleaning pipeline  
✅ **Indexing**: Created Lucene index with proper configuration  
✅ **Boolean Retrieval**: Implemented AND, OR, NOT queries  
✅ **Verification**: Logic checking ensures correct results  

## 🛠️ Interactive Mode

After running the test queries, enter interactive mode to try your own Boolean queries:

```
Enter query: cat AND mouse
🔍 Searching for: cat AND mouse
📄 Results: ['d1', 'd14']
📊 Total matches: 2
```

## 📚 References

- [Pyserini Documentation](https://github.com/castorini/pyserini)
- [Lucene Boolean Query Syntax](https://lucene.apache.org/core/documentation.html)
- [Porter Stemming Algorithm](https://tartarus.org/martin/PorterStemmer/)

## ✨ Features

- **Modular Design**: Separate modules for each component
- **Error Handling**: Robust error checking and reporting  
- **Verification**: Logic validation for query results
- **Interactive Mode**: Test custom Boolean queries
- **Detailed Logging**: Step-by-step process information
- **Comprehensive Reports**: Assignment deliverable generation