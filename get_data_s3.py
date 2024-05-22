import boto3
import os
from urllib.parse import urlparse
from dotenv import dotenv_values

def download_images_from_s3_folder(s3_url, local_dir):
    try:
        # Load environment variables from .env file
        variables = dotenv_values(".env")

        # Get AWS access key ID, secret access key, and region from environment variables
        access_key = variables.get("aws_access_key_id")
        secret_key = variables.get("aws_secret_access_key")
        region_name = variables.get("region_name")

        # Parse the S3 URL
        parsed_url = urlparse(s3_url)
        bucket_name = parsed_url.netloc
        folder_path = parsed_url.path.strip('/')

        # Initialize boto3 client with the provided AWS credentials
        s3_client = boto3.client('s3',
                                 region_name=region_name,
                                 aws_access_key_id=access_key,
                                 aws_secret_access_key=secret_key)

        # Ensure local directory exists
        os.makedirs(local_dir, exist_ok=True)

        # Paginator to list all objects in the specified folder (and its subfolders)
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=folder_path):
            for obj in page.get('Contents', []):
                object_key = obj['Key']
                if object_key.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    # Download the image file to the local directory
                    local_path = os.path.join(local_dir, os.path.basename(object_key))
                    s3_client.download_file(bucket_name, object_key, local_path)
                    print(f"Downloaded: {object_key} to {local_path}")

        print("All images downloaded successfully.")

    except Exception as e:
        print(f"Error downloading images: {e}")

# Example S3 URL and local directory
local_dir = "journal_no_2157"
s3_url = f"s3://logomatch/{local_dir}/images/"

# Call the function to download images from the S3 folder
download_images_from_s3_folder(s3_url, local_dir)
