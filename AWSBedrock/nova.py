import boto3
import json
import base64
import os

# Farklı görsel modelleri dene
models_to_try = [
    "stability.stable-diffusion-xl-v1",
    "amazon.titan-image-generator-v1",
    "amazon.titan-image-generator-v2:0"
]

prompt_data = "4k hd image of a beach, blue sky rainy season at bali view"

bedrock = boto3.client(service_name="bedrock-runtime", region_name='eu-west-2')

for model_id in models_to_try:
    try:
        print(f"\n🔄 Denenen model: {model_id}")
        
        if "stability" in model_id:
            # Stable Diffusion formatı
            payload = {
                "text_prompts": [{"text": prompt_data, "weight": 1}],
                "cfg_scale": 10,
                "seed": 0,
                "steps": 50,
                "width": 1024,
                "height": 1024
            }
        elif "titan" in model_id:
            # Amazon Titan formatı
            payload = {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {
                    "text": prompt_data,
                    "negativeText": "blurry, low quality"
                },
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "height": 1024,
                    "width": 1024,
                    "cfgScale": 8.0,
                    "seed": 0
                }
            }
        
        response = bedrock.invoke_model(
            body=json.dumps(payload),
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        
        response_body = json.loads(response.get("body").read())
        
        # Stable Diffusion yanıt formatı
        if "artifacts" in response_body:
            artifact = response_body["artifacts"][0]
            image_bytes = base64.b64decode(artifact["base64"])
        # Titan yanıt formatı
        elif "images" in response_body:
            image_bytes = base64.b64decode(response_body["images"][0])
        else:
            print(f"❌ Beklenmeyen yanıt formatı: {list(response_body.keys())}")
            continue
        
        # Görseli kaydet
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"{output_dir}/beach_{model_id.replace('.', '_').replace(':', '_')}.png"
        
        with open(file_name, "wb") as f:
            f.write(image_bytes)
        
        print(f"✅ Başarılı! Görsel kaydedildi: {file_name}")
        break
        
    except Exception as e:
        print(f"❌ {model_id} başarısız: {e}")

print("\n🏁 İşlem tamamlandı!")