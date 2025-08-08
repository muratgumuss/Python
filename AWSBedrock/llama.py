import boto3
import json

print("AWS Bedrock Llama 3 Test Başlıyor...")

try:
    client = boto3.client('bedrock-runtime', region_name='eu-west-2')
    print("✓ AWS client oluşturuldu")

    prompt = "Merhaba!"
    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.7,
        "top_p": 0.9
    }

    response = client.invoke_model(
        modelId='meta.llama3-8b-instruct-v1:0',
        body=json.dumps(body)
    )

    print("✓ Model yanıtı alındı")
    response_body = json.loads(response['body'].read())
    print("\n--- Llama 3 Yanıtı ---")
    print(response_body['generation'])

except Exception as e:
    print(f"✗ Hata: {e}")
    print("AWS credentials ve model erişimini kontrol edin")