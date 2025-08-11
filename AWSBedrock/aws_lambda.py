import boto3
import botocore.config
import json
from datetime import datetime

def blog_generate_using_bedrock(blogtopic: str) -> str:
    prompt = f"""<s>[INST]Human:
    Write a 200 words blog on the topic: {blogtopic}.
    Assistant:[/INST]
    """
    body = {
        "prompt": prompt,
        "max_gen_len": 200,
        "temperature": 0.5,
        "top_p": 0.9,
    }
    
    try:
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name="eu-west-2",
            config=botocore.config.Config(
                read_timeout=300,
                connect_timeout=60,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
        
        # invoke_model'den dönen response'u bir değişkene ata
        response = bedrock.invoke_model(
            body=json.dumps(body),
            modelId="meta.llama3-8b-instruct-v1:0",
            accept="application/json",
            contentType="application/json"
        )
        
        # Response'dan body'yi oku
        response_content = response.get("body").read()
        response_data = json.loads(response_content)
        
        # Llama modeli için generation field'ını kullan
        blog_details = response_data.get("generation", "")
        
        return blog_details
        
    except Exception as e:
        print(f"Error generating blog: {str(e)}")
        return f"Error: {str(e)}"

# Kullanım örneği
# blog_text = blog_generate_using_bedrock("Artificial Intelligence")
# print(blog_text)

def save_blog_details_to_s3(s3_key,s3_bucket, generate_blog):
    s3 = boto3.client('s3')
    try:
        s3.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=generate_blog.encode('utf-8')
        )
        print(f"Blog saved to S3 bucket {s3_bucket} with key {s3_key}")
    except Exception as e:
        print(f"Error saving blog to S3: {str(e)}")


def lambda_handler(event, context):
    # TODO implement
    event = json.loads(event['body'])
    blogtopic = event['blogtopic']
    generate_blog = blog_generate_using_bedrock(blogtopic=blogtopic)

    if generate_blog:
        current_time = datetime.now().strftime("%H%M%S")
        s3_key = f"blog_output/{current_time}.txt"
        s3_bucket = "aws-bedrock-mgm" 
        save_blog_details_to_s3(s3_key=s3_key, s3_bucket=s3_bucket, generate_blog=generate_blog)
    else:
        print("Blog generation failed.")

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Blog generation completed successfully.',
            'blog_content': generate_blog
        }), 
        'headers': {
            'Content-Type': 'application/json'
        }
    }
