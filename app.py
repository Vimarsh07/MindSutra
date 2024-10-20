import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.llms import CTransformers

# Cache loading of PDF documents
@st.cache_resource
def load_documents():
    pdf_loader = DirectoryLoader(
        path="/Users/Vimarsh/Desktop/MindSutra/data/PDF/", 
        glob="*.pdf", 
        loader_cls=PyPDFLoader
    )
    return pdf_loader.load()

document_data = load_documents()

# Split documents into manageable chunks
text_chunk_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # Increased chunk size for better context understanding
    chunk_overlap=64  # Small overlap to preserve context
)
document_chunks = text_chunk_splitter.split_documents(document_data)

# Cache embedding generation and vector store creation
@st.cache_resource
def create_vector_store():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return FAISS.from_documents(document_chunks, embedding_model)

vector_database = create_vector_store()

# Create language model instance
@st.cache_resource
def load_language_model():
    return CTransformers(
        model="/Users/Vimarsh/Desktop/MindSutra/models/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config={'max_new_tokens': 128, 'temperature': 0.7}  # Adjusted temperature for more balanced responses
    )

language_model = load_language_model()

# Memory to maintain conversation context
chat_memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True)

# Set up the conversational retrieval chain
@st.cache_resource
def get_conversational_chain():
    return ConversationalRetrievalChain.from_llm(
        llm=language_model,
        chain_type='stuff',
        retriever=vector_database.as_retriever(search_kwargs={"k": 3}),  # Increased k for more relevant context
        memory=chat_memory
    )

conversational_chain = get_conversational_chain()

# Streamlit app interface
def main():
    st.title("AI Therapy Assistant")

    # Initialize session state
    if "conversation_log" not in st.session_state:
        st.session_state["conversation_log"] = []
    if "responses" not in st.session_state:
        st.session_state["responses"] = ["Hi there! Feel free to ask me anything."]
    if "user_prompts" not in st.session_state:
        st.session_state["user_prompts"] = ["Hello!"]

    # Display conversation
    display_conversation_interface()

def handle_user_query(user_query):
    response = conversational_chain({"question": user_query, "conversation_history": st.session_state['conversation_log']})
    st.session_state["conversation_log"].append((user_query, response["answer"]))
    return response["answer"]

def display_conversation_interface():
    chat_container = st.container()
    response_container = st.container()

    with chat_container:
        with st.form(key="input_form", clear_on_submit=True):
            user_input = st.text_input("Your Question:", placeholder="What's on your mind today?")
            submit_button = st.form_submit_button(label="Send")

            if submit_button and user_input.strip():
                with st.spinner("Thinking..."):
                    bot_response = handle_user_query(user_input)
                    st.session_state['user_prompts'].append(user_input)
                    st.session_state['responses'].append(bot_response)
            elif submit_button and not user_input.strip():
                st.warning("Please enter a valid question.")

        if st.session_state['responses']:
            with response_container:
                for idx in range(len(st.session_state['responses'])):
                    message(st.session_state['user_prompts'][idx], is_user=True, key=f"{idx}_user", avatar_style="thumbs")
                    message(st.session_state['responses'][idx], key=str(idx), avatar_style="fun-emoji")

    if st.button("Clear Conversation"):
        st.session_state['conversation_log'] = []
        st.session_state['user_prompts'] = ["Hello!"]
        st.session_state['responses'] = ["Hi there! Feel free to ask me anything."]

    if st.button("Save Conversation"):
        with open("conversation.txt", "w") as f:
            for idx in range(len(st.session_state['responses'])):
                f.write(f"User: {st.session_state['user_prompts'][idx]}\n")
                f.write(f"Bot: {st.session_state['responses'][idx]}\n\n")
        st.success("Conversation saved successfully.")

if __name__ == "__main__":
    main()
