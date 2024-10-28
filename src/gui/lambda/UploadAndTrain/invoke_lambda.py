import boto3
import json
import sys

def invoke_lambda_function(function_name, bucket_name, image_key):
    lambda_client = boto3.client('lambda')
    payload = {
        'bucket_name': bucket_name,
        'image_key': image_key
    }
    try:
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',  # For synchronous execution
            Payload=json.dumps(payload)
        )
        response_payload = response['Payload'].read()
        print("Lambda function response:", response_payload.decode())
    except Exception as e:
        print(f"Error invoking Lambda function: {e}")

if __name__ == "__main__":
    # Usage: python invoke_lambda.py <function_name> <bucket_name> <image_key>
    if len(sys.argv) < 4:
        print("Usage: python invoke_lambda.py <function_name> <bucket_name> <image_key>")
    else:
        function_name = sys.argv[1]
        bucket_name = sys.argv[2]
        image_key = sys.argv[3]
        invoke_lambda_function(function_name, bucket_name, image_key)
