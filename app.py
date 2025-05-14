import streamlit as st
from csv_handler import CSVHandler
from rag_system import RAGSystem
from utils import AnswerFormatter, generate_data_summary
import tempfile
import os
import traceback


# Initialize session state
def init_session():
    if 'chat' not in st.session_state:
        st.session_state.chat = {
            'handler': CSVHandler(),
            'rag': RAGSystem(),
            'messages': [],
            'file_key': None,
            'formatter': AnswerFormatter()
        }


init_session()

# UI Configuration
st.set_page_config(page_title="CSV Analyst", layout="wide")
st.title("🔍 CSV Data Analyst")
st.caption("Upload a CSV file and ask questions about your data")

# File Upload Section
with st.sidebar:
    uploaded_file = st.file_uploader(
        "Choose CSV",
        type="csv",
        help="Maximum file size: 200MB",
        key="file_uploader"
    )

    if st.button("Reset Chat", type="primary"):
        st.session_state.clear()
        init_session()
        st.rerun()

# File Processing Logic
if uploaded_file and uploaded_file != st.session_state.chat.get('file_key'):
    with st.spinner("Analyzing your data..."):
        try:
            # Reset previous state
            st.session_state.chat['messages'] = []
            st.session_state.chat['file_key'] = uploaded_file

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Process CSV
            if st.session_state.chat['handler'].load_csv(tmp_path):
                df = st.session_state.chat['handler'].df

                # Initialize RAG
                summary = st.session_state.chat['handler'].get_summary()
                documents = st.session_state.chat['rag'].generate_documents_from_data(df, summary)
                st.session_state.chat['rag'].prepare_vectorstore(documents)
                st.session_state.chat['rag'].create_qa_chain()

                # Add success message
                welcome_msg = generate_data_summary(summary)
                st.session_state.chat['messages'].append(
                    {"role": "assistant", "content": f"✅ Data loaded successfully!\n\n{welcome_msg}"}
                )

            # Clean up
            os.unlink(tmp_path)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.session_state.chat['messages'].append(
                {"role": "assistant", "content": "❌ Failed to process file. Please try another CSV."}
            )

# Chat Display
for msg in st.session_state.chat.get('messages', []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask about your data..."):
    try:
        # Add user message
        st.session_state.chat['messages'].append({"role": "user", "content": prompt})

        # Generate response
        if st.session_state.chat['handler'].df is not None:
            with st.spinner("Thinking..."):
                answer = st.session_state.chat['rag'].answer_question(prompt)
                formatted = st.session_state.chat['formatter'].format_response(answer, prompt)
                st.session_state.chat['messages'].append(
                    {"role": "assistant", "content": formatted}
                )
        else:
            st.session_state.chat['messages'].append(
                {"role": "assistant", "content": "Please upload a CSV file first"}
            )

        st.rerun()
    except Exception as e:
        st.error(f"Error processing question: {traceback.format_exc()}")
        st.session_state.chat['messages'].append(
            {"role": "assistant", "content": f"⚠️ Error: {str(e)}"}
        )
        st.rerun()

# Debug Panel (Collapsible)
with st.expander("Developer Tools", expanded=False):
    if st.session_state.chat['handler'].df is not None:
        st.download_button(
            "Download Processed Data",
            st.session_state.chat['handler'].df.to_csv(index=False),
            file_name="analyzed_data.csv"
        )
    st.json(st.session_state.chat.get('messages', []), expanded=False)


    