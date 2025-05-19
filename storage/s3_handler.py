import boto3
import logging
from botocore.exceptions import ClientError
import os

class S3Handler:
    def __init__(self, config):
        self.config = config

        if self.config.s3_enabled:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=config.s3_key,
                    aws_secret_access_key=config.s3_secret,
                    endpoint_url=config.s3_endpoint,
                    region_name='gra'  # OVH Gravelines region
                )
                logging.info("S3 client initialized.")
            except Exception as e:
                logging.error(f"Error initializing S3 client: {str(e)}")
                self.s3_client = None
        else:
            self.s3_client = None
            logging.info("S3 is disabled in config.")

    async def upload_file(self, file_path, object_name=None):
        """
        Upload a file to S3 in the 'OCTOVAR/' folder.
        Returns the public URL if successful, else None.
        """
        if not self.config.s3_enabled or self.s3_client is None:
            logging.warning("S3 upload skipped (disabled or client unavailable).")
            return None

        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return None

        # Use filename if object_name not specified
        if object_name is None:
            object_name = file_path.split('/')[-1]

        # Upload under 'OCTOVAR/' prefix
        object_name = f"OCTOVAR/{object_name}"

        try:
            self.s3_client.upload_file(
                file_path,
                self.config.s3_bucket,
                object_name,
                ExtraArgs={'ACL': 'public-read',
                'ContentType':'video/mp4'}  # Set content type for video files
            )

            # Construct the public URL
            public_url = f"https://{self.config.s3_bucket}.{self.config.s3_public_endpoint}/{object_name}"
            logging.info(f"File uploaded successfully to {public_url}")
            return public_url

        except ClientError as e:
            logging.error(f"Error uploading file to S3: {str(e)}")
            return None
