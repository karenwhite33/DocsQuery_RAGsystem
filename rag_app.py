import sys
if "torch._classes" in sys.modules:
    del sys.modules["torch._classes"]
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings

# --- Load Local Model ---
llm = Llama(
    model_path="./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf",
    n_ctx=512,
    n_threads=6,
    n_gpu_layers=30
)

# --- Embeddings & Vectorstore ---
embedding_model = SentenceTransformer(r"D:\AI Bootcamp Github\RAG\models\e5-base-v2", device="cuda")
embedding_function = SentenceTransformerEmbeddings(model_name=r"D:\AI Bootcamp Github\RAG\models\e5-base-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_function, allow_dangerous_deserialization=True)

# --- Retriever with Compression Filter ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
compressor = EmbeddingsFilter(embeddings=embedding_function, similarity_threshold=0.65)
retriever_with_filter = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# --- Prompt Template ---
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the user's question using the documentation provided.

Context:
{context}

Question: {question}
Answer (only to the question above):"""
)

# --- Ask Function ---
def ask_mistral(query, retriever_with_filter, top_k=5, max_new_tokens=512):
    try:
        results = retriever_with_filter.invoke(query)
        if not results:
            return "No relevant documents found.", [], []

        context = "\n\n".join([doc.page_content for doc in results])
        prompt = prompt_template.format(context=context, question=query)

        response = llm(
            prompt,
            max_tokens=300,
            temperature=0,
            stop=["\nQuestion:"],
            echo=False,
        )
        answer = response["choices"][0]["text"].strip()

        # Collect unique domains and doc titles
        domains = list({doc.metadata.get("domain") for doc in results if "domain" in doc.metadata})
        titles = list({doc.metadata.get("source") for doc in results if "source" in doc.metadata})

        return answer if answer else "No answer generated.", domains, titles

    except Exception as e:
        return f"Error: {str(e)}", [], []



# --- Streamlit UI ---
st.set_page_config(page_title="DocsQuery", layout="centered")
st.title("🧠 DocsQuery - RAG")
st.markdown("""
Your assistant can answer questions based on internal official documents across the following areas:

- 🏥 **VA (Veterans Affairs)**
- 🧾 **SSA (Social Security Administration)**
- 🚗 **DMV (Department of Motor Vehicles)**
- 🎓 **StudentAid (Federal Student Aid)**
""")


query = st.text_input("Enter your question:", placeholder="e.g. What are the motor vehicle laws?")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        answer, domains, titles = ask_mistral(query, retriever_with_filter)

        st.markdown("#### 📬 Answer:")
        st.markdown(answer.replace("\n", "\n\n"))

        if domains:
            st.markdown(f"📂 **Source domain(s):** {', '.join(domains)}")
        if titles:
            st.markdown("📄 **Document titles used:**")
            for title in titles:
                st.markdown(f"- {title}")





