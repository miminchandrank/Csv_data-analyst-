from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
import numpy as np
import os


class RAGSystem:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Initialize LLM
        model_path = "C:\csv-chatbot\orca-mini-3b.ggmlv3.q4_0.bin"  # This should be downloaded separately
        self.llm = GPT4All(
            model=model_path,
            max_tokens=2048,
            temp=0.7,
            backend='gptj',
            verbose=False
        )

        self.vectorstore = None
        self.qa_chain = None

    def prepare_vectorstore(self, documents: List[str]):
        """Create FAISS vectorstore from documents"""
        self.vectorstore = FAISS.from_texts(
            texts=documents,
            embedding=self.embedding_model
        )

    def create_qa_chain(self):
        """Create QA chain with custom prompt"""
        prompt_template = """Use the following context to answer the question at the end. 
        The context contains information about a CSV dataset. If you don't know the answer, 
        just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def generate_documents_from_data(self, df: pd.DataFrame, summary: Dict[str, Any]) -> List[str]:
        """Generate text documents from data for vector storage"""
        documents = []

        # Add metadata summary
        metadata_doc = f"Dataset Metadata:\nShape: {summary['metadata']['shape']}\nColumns: {', '.join(summary['metadata']['columns'])}\n"
        documents.append(metadata_doc)

        # Add column details
        for col, stats in summary['analysis']['column_stats'].items():
            col_doc = f"Column: {col}\n"
            col_doc += f"Data type: {summary['metadata']['dtypes'][col]}\n"
            col_doc += f"Unique values: {stats['unique_count']}\n"
            if stats['is_constant']:
                col_doc += "This column has a constant value.\n"
            if stats['most_frequent'] is not None:
                col_doc += f"Most frequent value: {stats['most_frequent']}\n"
            if stats['max_length'] is not None:
                col_doc += f"Max length: {stats['max_length']}\n"
            documents.append(col_doc)

        # Add missing values info
        missing_doc = "Missing Values Analysis:\n"
        for col, count in summary['analysis']['missing_values']['count_by_column'].items():
            if count > 0:
                missing_doc += f"{col}: {count} missing values ({summary['analysis']['missing_values']['percentage_by_column'][col]:.2f}%)\n"
        documents.append(missing_doc)

        # Add sample data
        sample_doc = "Sample Data (first 5 rows):\n"
        sample_doc += df.head().to_string()
        documents.append(sample_doc)

        return documents

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using the RAG system"""
        if not self.qa_chain:
            return {"answer": "System not initialized", "source_documents": []}

        result = self.qa_chain({"query": question})
        return {
            "answer": result['result'],
            "source_documents": [doc.page_content for doc in result['source_documents']]
        }