import time
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class FastMTCNN:
    def __init__(self):
        self.detector = MTCNN(device=device)

    def extract_face(self, image: Image.Image, required_size=(160, 160)) -> tuple[torch.Tensor, Image.Image, Image.Image]:
        """
        Extrae y procesa un rostro de una imagen de entrada.\n
        Este método detecta rostros en la imagen proporcionada, recorta el primer rostro detectado,
        y lo procesa para reconocimiento facial convirtiéndolo a un tensor y redimensionándolo.
 
        Args:
            image (Image.Image): Imagen PIL de entrada que contiene uno o más rostros
            required_size (tuple[int, int], opcional): Tamaño objetivo para redimensionar la imagen del rostro.
                Por defecto es 160x160.
        
        Returns:
            tuple[torch.Tensor, Image.Image, Image.Image]: Una tupla que contiene:
            - face_tensor: Rostro procesado como tensor de PyTorch
            - face: Rostro recortado original como imagen PIL
            - final_image: Rostro redimensionado como imagen PIL según required_size
        
        Raises:
            ValueError: Si no se detectan rostros en la imagen de entrada.
        
        Notas:
            - La imagen primero se convierte a formato RGB
            - La imagen puede ser reducida si es demasiado grande usando _limit_image_size
            - Solo se procesa el primer rostro detectado
            - El tensor de salida se mueve al dispositivo predeterminado (CPU/GPU)
        """

        image = image.convert("RGB")
        
        scaled_image, scaling_factor_x, scaling_factor_y = self._limit_image_size(image)
        # scaled_image, scaling_factor_x, scaling_factor_y = image, 1.0, 1.0

        face_bounding_boxes, _ = self.detector.detect(scaled_image)

        if face_bounding_boxes is None or len(face_bounding_boxes) == 0:
            raise ValueError("No se detectaron rostros en la imagen.")

        x1, y1, x2, y2 = [int(b) for b in face_bounding_boxes[0]]
        
        x1 = int(x1 * scaling_factor_x)
        y1 = int(y1 * scaling_factor_y)
        x2 = int(x2 * scaling_factor_x)
        y2 = int(y2 * scaling_factor_y)
        
        face = image.crop((x1, y1, x2, y2))

        transform = transforms.Compose([
            transforms.Resize(required_size),
            transforms.ToTensor(),
        ])

        face_tensor = transform(face).unsqueeze(0).to(device)
        final_image = face.resize(required_size)

        return face_tensor, face, final_image

    def _limit_image_size(self, image: Image.Image) -> tuple[Image.Image, float, float]:
        """
        Redimensiona la imagen para limitar dimensiones manteniendo la proporción.\n
        Este método reduce las imágenes que exceden las dimensiones máximas (640 de alto o 480 de ancho)
        mientras preserva su proporción de aspecto original.\n
        - Para imágenes horizontales, la altura se limita a 640px.
        - Para imágenes verticales, el ancho se limita a 480px.
        
        Args:
            image (Image.Image): La imagen PIL de entrada a redimensionar.
       
        Returns:
            tuple[Image.Image, float, float]: Una tupla que contiene:
            - La imagen PIL redimensionada
            - Factor de escala en eje X (ancho_original/nuevo_ancho)
            - Factor de escala en eje Y (alto_original/nuevo_alto)
    
        Nota:
            Las imágenes más pequeñas que las dimensiones máximas se devuelven sin cambios con factores de escala de 1.0.
        """
        
        resized_image = image
        max_height, max_width = 640, 480

        if image.width > image.height:
            if image.height > max_height:
                aspect_ratio = image.width / image.height
                new_width = int(max_height * aspect_ratio)
                resized_image = image.resize((new_width, max_height))
        
        else:
            if image.width > max_width:
                aspect_ratio = image.height / image.width
                new_height = int(max_width * aspect_ratio)
                resized_image = image.resize((max_width, new_height))
        
        original_width, original_height = image.size
        scaled_width, scaled_height = resized_image.size

        scaling_factor_x = original_width / scaled_width
        scaling_factor_y = original_height / scaled_height

        return resized_image, scaling_factor_x, scaling_factor_y
    
class FaceEmbeddingModel:
    def __init__(self):
        self.detector = FastMTCNN()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def extract_face(self, image: Image.Image, required_size=(160, 160)) -> tuple[torch.Tensor, Image.Image, Image.Image, float]:
        start_time = time.time()
        face_tensor, face_image, final_image = self.detector.extract_face(image, required_size)
        processing_time = time.time() - start_time
    
        return face_tensor, face_image, final_image, processing_time

    def get_embedding(self, face_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
        start_time = time.time()
        embedding = self.resnet(face_tensor).detach().cpu().numpy().flatten()
        processing_time = time.time() - start_time

        return embedding, processing_time