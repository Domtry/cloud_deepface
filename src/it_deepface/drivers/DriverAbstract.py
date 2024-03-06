from typing import Union


class DriverAbstract:
    
    def __init__(self,
        region:str, 
        hostname:str, 
        access_key:str, 
        secret_key:str, 
        secure:bool) -> None:
        self.client = None
    
    def load_data(self) -> list[str]:
        raise NotImplementedError()
    
    
    def upload_representation(self, filename: str, data: bytes, bucket_name: str) -> None:
        raise NotImplementedError()
    
    
    def get_representation(self, filename: str, bucket_name: str) -> Union[bytes, None]:
        raise NotImplementedError()