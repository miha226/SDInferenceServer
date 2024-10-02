import os
import numpy as np
import torch
from PIL import Image
import io
from diffusers import StableDiffusionPipeline
from triton_python_backend_utils import Tensor, InferenceResponse, InferenceRequest, get_input_tensor_by_name, triton_string_to_numpy

class TritonPythonModel:

    def initialize(self, args):
        # Get model repository path
        model_repository = args['model_repository']
        model_version = args['model_version']
        model_path = os.path.join(model_repository, model_version, "sdxl_turbo")

        # Load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(self.device)

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get the image input
            image_tensor = get_input_tensor_by_name(request, "IMAGE")
            image_bytes = image_tensor.as_numpy()

            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes.tobytes())).convert("RGB")

            # Get the prompt input
            prompt_tensor = get_input_tensor_by_name(request, "PROMPT")
            prompt = prompt_tensor.as_numpy()[0].decode('utf-8')

            # Preprocess the image if necessary
            # For example, resize or normalize

            # Run inference
            with torch.no_grad():
                output_image = self.pipeline(prompt=prompt, image=image).images[0]

            # Convert the output image to bytes
            output_buffer = io.BytesIO()
            output_image.save(output_buffer, format='JPEG')
            output_bytes = np.frombuffer(output_buffer.getvalue(), dtype=np.uint8)

            # Create the output tensor
            output_tensor = Tensor("OUTPUT_IMAGE", output_bytes)

            # Create the inference response
            inference_response = InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses
