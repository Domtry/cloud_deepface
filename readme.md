#Cloud Deepface 

cloud_deepface
## Installation

download package [pip](https://github.com/Domtry/cloud_deepface/blob/main/dist/it_deepface-0.0.1.tar.gz.)
and copy file in your racine projet.

```bash
pip install cloud_deepface  
```

## Usage

```python
from cloud_deepface import DeepFace


#config minio
from cloud_deepface.driver import config

config.DRIVER_HOSTNAME = "driver hostname"
config.DRIVER_SECRET_KEY = "driver secret key"
config.DRIVER_ACCESS_KEY = "driver access key"
config.DRIVER_BUCKET_NAME = "cloud bucket storage name"
config.DRIVER_REGION = "driver region"
config.DRIVER_PROTOCOL_SECURE = False

# select image url
got_url = "https://www.shutterstock.com/image-photo/image-african-man-foot-wearworkshop-600nw-1839501016.jpg"

# lunch detection image url
response = DeepFace.find_from_cloud(
    img_url=got_url,
    synchronization=False,
    bucket_name="storages",
    driver="minio",
    bin_path="./datasets")

# execute verification
fdd_response = FDD.verify_images(path_img1, path_img2)


# returns '{'verified': True, 'path_img': ['link'], 'threshold': 0.68}'
print(fdd_response)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
