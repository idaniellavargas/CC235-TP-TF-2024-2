import gradio as gr
from PIL import Image
import numpy as np
from facenet import FaceEmbeddingModel
from search_face import query_face_embedding

model = FaceEmbeddingModel()

def process_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    
    face_tensor, _, imagen_final, tiempo_procesamiento_extraccion = model.extract_face(image)
    face_embedding, tiempo_procesamiento_embedding = model.get_embedding(face_tensor)
    
    matches = query_face_embedding(face_embedding, k=5, test=True)
    resultados = [[match["metadata"]["student_code"], match["score"]] for match in matches["matches"]]
    
    tiempos = f"- Extracción de rostro: {tiempo_procesamiento_extraccion:.3f} s\n- Cálculo de embedding: {tiempo_procesamiento_embedding:.3f} s"
    
    return imagen_final, resultados, tiempos

output_components = [
    gr.Image(type="pil", label="Imagen final (160x160)"),
    gr.DataFrame(headers=["student_code", "score"], label="Matches"),
    gr.Textbox(label="Tiempos de procesamiento"),
]

demo = gr.Interface(fn=process_image, inputs="image", outputs=output_components)
demo.launch(share=True, server_name="0.0.0.0", server_port=7860)