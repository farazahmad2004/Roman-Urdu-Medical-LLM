from huggingface_hub import login
# login(token="")
import streamlit as st
import time
import torch
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

st.set_page_config(page_title="AI Doctor (Roman Urdu)", page_icon="🩺", layout="centered")

@st.cache_resource
def load_system():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base_model, "farazahmad2004/NLP-Medical-Chatbot-Llama-1B")

    def load_docs(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            raw_chunks = text.split("###")
            return [Document(page_content=chunk.strip(), metadata={"source": filepath}) for chunk in raw_chunks if chunk.strip()]
        except:
            return []

    docs = load_docs("./datasets/RAG_dataset/rag_dataset.txt") 

    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vector_db = Chroma.from_documents(docs, embedding_model)
    semantic_retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.9, 0.1] 
    )
    
    return model, tokenizer, ensemble_retriever

model, tokenizer, retriever = load_system()

def ask_doctor(question, use_rag):
    context_text = ""
    
    if use_rag and retriever:
        docs = retriever.invoke(question)
        context_text = docs[0].page_content
        
        rag_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert Medical AI. Answer the question using ONLY the provided Context.
Rules:
1. Act as a professional Doctor.
2. Do NOT talk about yourself. Do NOT say "Main" (I).
3. Speak in clear Roman Urdu.
4. If Context is unrelated, say "Maaf kijiye, maloomat nahi."

CONTEXT:
{context_text}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    else:
        rag_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Answer as a Doctor in Roman Urdu.<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    inputs = tokenizer(rag_prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.3, 
        repetition_penalty=1.2,
        do_sample=True
    )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in response_text:
        response_text = response_text.split("assistant")[-1].strip()
        
    return response_text, context_text

st.title("🩺 Roman Urdu AI Doctor")
st.caption("Running locally on CPU using Llama-1B & Hybrid RAG")

with st.sidebar:
    st.header("Settings")
    use_rag = st.toggle("Enable RAG", value=True)
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Sawall likhein..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Soch raha hoon..."):
            response, ctx = ask_doctor(prompt, use_rag)
            message_placeholder.markdown(response)
            
            if use_rag and ctx:
                with st.expander("Show Medical Context"):
                    st.info(ctx)
    
    st.session_state.messages.append({"role": "assistant", "content": response})