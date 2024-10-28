import json
import boto3
import base64
from botocore.exceptions import ClientError

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    # Assuming the image is sent in the body as a base64 string
    body = json.loads(event['body'])
    image_data = body['image']  # Base64 string
    image_name = body['name']  # Image name (optional)

    # Decode the image
    image_bytes = base64.b64decode(image_data)

    # Save to S3
    bucket_name = 'demo-image-upload-bucket'
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=image_name,
            Body=image_bytes,
            # ContentType='image/jpeg'  # Adjust as needed
        )
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Image uploaded successfully!'})
        }
    except ClientError as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'message': str(e)})
        }
