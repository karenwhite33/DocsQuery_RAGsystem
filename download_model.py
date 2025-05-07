from sentence_transformers import SentenceTransformer
import shutil
import os

# Ruta donde se va a guardar
save_path = "D:/AI Bootcamp Github/RAG/models/e5-base-v2"

# Si ya existe, la eliminamos completamente para evitar errores de descarga incompleta
if os.path.exists(save_path):
    print(f"Deleting existing folder: {save_path}")
    shutil.rmtree(save_path)

# Descargar modelo desde Hugging Face
print("⬇Downloading model 'intfloat/e5-base-v2'...")
model = SentenceTransformer("intfloat/e5-base-v2")

# Guardar modelo localmente
print(f"Saving model to: {save_path}")
model.save(save_path)

print("✅ Model downloaded and saved successfully.")
