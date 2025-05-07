# 🧠 DocsQuery — Local RAG Assistant with Mistral & FAISS

A fully local Retrieval-Augmented Generation (RAG) assistant that answers questions exclusively from official internal documentation. No API calls, no hallucinations — 100% private and offline.

## 🚨 Problem

Public-facing LLMs often produce fabricated answers, require constant internet access, or expose sensitive data. I needed a tool to query internal government documents (SSA, DMV, VA, StudentAid) **without leaking context or relying on cloud APIs**. Most tutorials failed with real-world docs or broke on local deployments.

## ✅ Solution

**DocsQuery** is a local-first, production-ready RAG app that runs entirely on your machine using:

- 🔍 **FAISS** for semantic vector search
- 🤖 **Mistral 7B Instruct (GGUF)** running via `llama.cpp` (no GPU dependency required)
- 🧠 **e5-base-v2** embeddings for retrieval and compression filtering
- 🧊 **LangChain compression retriever** with similarity filtering
- 🌐 **Streamlit UI** for intuitive interface and live testing
- 🌍 **ngrok** for lightweight external access without cloud deployment

- 👉 [DEMO](https://drive.google.com/file/d/1PBuibjFN0Jj-gzQDTQECQE-BMTYOJbmx/view?usp=drive_link)

- ![doscquery_screen](https://github.com/user-attachments/assets/2e723cdb-2106-4860-a221-08602018e46b)

- ![doscquery_screen2](https://github.com/user-attachments/assets/b9ec1d6e-d9c1-46da-b959-1c482d4bcbe0)

## 💻 Tech Stack

- Python
- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- SentenceTransformers (`intfloat/e5-base-v2`)
- Streamlit
- Ngrok (optional live deploy)

## ⚙️ Features

- 🔐 **Offline, private and fast**: Nothing leaves your machine
- 📚 **Contextual Compression**: Only the most relevant text is passed to the model
- 🔎 **Filtered Retrieval**: Similarity threshold avoids irrelevant or noisy inputs
- 📑 **Sources displayed**: Each answer includes the domain + document titles used
- 🔄 **Live UI** with Streamlit
- 🧪 **Fine-tuned prompt for clear, precise answers**

## ⚠️ Challenges Overcome

- Manual download and resolution of `intfloat/e5-base-v2` to avoid broken transformers meta-tensor errors
- Inconsistent Hugging Face token behavior and authorization conflicts
- Streamlit and PyTorch watcher crash on C++ extensions (resolved with custom overrides)
- Controlled inference latency for Mistral with optimal `n_ctx`, `n_gpu_layers`, and prompt formatting

## 🚀 How to Run

1. Clone the repo and install requirements (see `env.yaml`)
2. Download models:
   - Mistral 7B GGUF from [TheBloke](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
   - e5-base-v2 via `SentenceTransformer("intfloat/e5-base-v2").save(...)`
3. Prebuild your FAISS index using your internal documents
4. Run the Streamlit app:

```bash
streamlit run rag_app.py
