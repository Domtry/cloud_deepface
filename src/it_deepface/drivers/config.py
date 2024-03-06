from it_deepface.drivers.DriverAbstract import DriverAbstract as Driver
from it_deepface.drivers.Minio import MinioClient


DRIVER_HOSTNAME = "driver hostname"
DRIVER_SECRET_KEY = "driver secret key"
DRIVER_ACCESS_KEY = "driver access key"
DRIVER_BUCKET_NAME = "cloud bucket storage name"
DRIVER_REGION = "driver region"
DRIVER_PROTOCOL_SECURE = False


def load_driver(driver_name:str) -> Driver :
    
    if driver_name is "minio":
        return MinioClient(
            region=DRIVER_REGION,
            hostname=DRIVER_HOSTNAME,
            access_key=DRIVER_ACCESS_KEY,
            secure=DRIVER_PROTOCOL_SECURE,
            secret_key=DRIVER_SECRET_KEY)
    
    raise Exception("driver not found, please ...")