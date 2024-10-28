import boto3
import sys

def upload_image_to_s3(file_name, bucket_name, object_name=None):
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, bucket_name, object_name or file_name)
        print(f"Successfully uploaded {file_name} to {bucket_name}/{object_name or file_name}")
    except Exception as e:
        print(f"Error uploading file: {e}")

if __name__ == "__main__":
    # Usage: python upload_image.py <file_name> <bucket_name> <object_name>
    if len(sys.argv) < 3:
        print("Usage: python upload_image.py <file_name> <bucket_name> [object_name]")
    else:
        file_name = sys.argv[1]
        bucket_name = sys.argv[2]
        object_name = sys.argv[3] if len(sys.argv) > 3 else None
        upload_image_to_s3(file_name, bucket_name, object_name)
