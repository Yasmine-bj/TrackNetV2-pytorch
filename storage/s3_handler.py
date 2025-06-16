import boto3
import logging
from botocore.exceptions import ClientError
import os
import io
import tempfile
import ffmpeg

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
                    region_name='gra' # région paramétrable
                )
                logging.info("S3 client initialized.")
            except Exception as e:
                logging.error(f"Error initializing S3 client: {e}")
                self.s3_client = None
        else:
            self.s3_client = None
            logging.info("S3 is disabled in config.")

    def process_video_with_ffmpeg(self, input_path: str) -> bytes:
        """
        Réencode la vidéo en H.264/AAC avec ffmpeg, 
        pour compatibilité streaming, puis renvoie les bytes.
        """
        temp_file = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
        try:
            (
                ffmpeg
                .input(input_path)
                .output(
                    temp_file,
                    **{
                        'c:v': 'libx264',
                        'profile:v': 'high',
                        'preset': 'fast',
                        'movflags': '+faststart',
                        'c:a': 'aac',
                        'b:a': '128k'
                    }
                )
                .overwrite_output()
                .run(quiet=True)
            )
            with open(temp_file, 'rb') as f:
                return f.read()
        except ffmpeg.Error as e:
            logging.error(f"ffmpeg error: {e.stderr.decode() if e.stderr else e}")
            raise
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def upload_file(self, file_path, object_name=None):
        """
        Réencode d'abord la vidéo, puis upload sur S3 depuis un BytesIO.
        Retourne l'URL publique en cas de succès, sinon None.
        """
        if not self.config.s3_enabled or not self.s3_client:
            logging.warning("S3 upload skipped (disabled or client unavailable).")
            return None

        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return None

        # Nom de l'objet
        if object_name is None:
            object_name = os.path.basename(file_path)
        object_name = f"OCTOVAR/{object_name}"

        try:
            # Réencodage vidéo
            logging.info(f"Processing video {file_path} with ffmpeg...")
            video_bytes = self.process_video_with_ffmpeg(file_path)
            buffer = io.BytesIO(video_bytes)

            # Upload depuis le buffer mémoire
            self.s3_client.upload_fileobj(
                Fileobj=buffer,
                Bucket=self.config.s3_bucket,
                Key=object_name,
                ExtraArgs={
                    'ACL': 'public-read',
                    'ContentType': 'video/mp4'
                }
            )

            public_url = f"https://{self.config.s3_bucket}.{self.config.s3_public_endpoint}/{object_name}"
            logging.info(f"File uploaded successfully to {public_url}")
            return public_url

        except ClientError as e:
            logging.error(f"Error uploading file to S3: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error during upload: {e}")
            return None
