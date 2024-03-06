import io
import os
from typing import Union
from datetime import timedelta

from minio import Minio, S3Error

from cloud_deepface.drivers.DriverAbstract import DriverAbstract


class MinioClient(DriverAbstract):
        
    def __init__(
        self, 
        hostname:str, 
        access_key:str, 
        secret_key:str, 
        region:str, 
        secure:bool=False):
                
        self.client = Minio(
            hostname,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
            secure=secure
        )
        
        self.bucket_name = "cloud-deepface-bucket"
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)
        
    
    def load_data(self, bucket_name:str) -> list[str]:
        
        found = self.client.bucket_exists(bucket_name)
        if not found:
            self.client.make_bucket(bucket_name)
        urls:list[str] = []
        
        bucket_object = self.client.list_objects(bucket_name, recursive=True)
        for item_obj in bucket_object:
            file_extension = os.path.splitext(item_obj.object_name)[1]
            if item_obj.size > 0 and file_extension.lower() in (".png", ".jpg", ".jpeg"):
                url = self.client.presigned_get_object(
                    bucket_name, item_obj.object_name, expires=timedelta(seconds=3600))
                urls.append(url)
        return urls
    
    
    def upload_representation(self, filename: str, data: bytes) -> None:
        byte_object = io.BytesIO(data)
        self.client.put_object(
            self.bucket_name, filename, byte_object, length=len(data))
        
    
    def get_representation(self, filename:str) -> Union[bytes, None]:
        try:
            response = self.client.get_object(self.bucket_name, filename)
            return response.read()
        
        except S3Error as err:
            return None
        except Exception as err:
            raise err
        
    
    