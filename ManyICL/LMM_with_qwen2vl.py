from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import time

class Qwen2VLAPI:
    def __init__(
        self,
        model="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
        img_token="<<IMG>>",
        seed=66,
        temperature=0.7,
        device="cuda" if torch.cuda.is_available() else "cpu",
        detail="auto"
    ):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model, 
            torch_dtype="auto", 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model)
        self.img_token = img_token
        self.seed = seed
        self.temperature = temperature
        self.device = device
        self.response_times = []
        self.token_usage = (0, 0, 0)

    def process_image(self, image_path):
        """Process image for model input"""
        if isinstance(image_path, str):
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            image = image_path
        return image

    def __call__(
        self,
        prompt,
        image_paths=[],
        real_call=True,
        count_time=False,
        max_tokens=50,
        content_only=True,
    ):
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        # Prepare messages in the required format
        messages = []
        images = []
        
        for image_path in image_paths:
            image = self.process_image(image_path)
            images.append(image)

        # Create message with images and text
        message_content = []
        for image in images:
            message_content.append({
                "type": "image",
                "image": image
            })
        message_content.append({
            "type": "text",
            "text": prompt
        })

        messages.append({
            "role": "user",
            "content": message_content
        })

        if not real_call:
            return {"messages": messages}

        start_time = time.time()

        # Prepare inputs for the model
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate response
        torch.manual_seed(self.seed)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=self.temperature
        )
        
        # Process output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        end_time = time.time()
        self.response_times.append(end_time - start_time)

        if content_only:
            return response
        else:
            return {
                "response": response,
                "messages": messages,
                "image_paths": image_paths,
                "time": end_time - start_time
            }