# IncubTek Image Duplicate Detection

it_deepface
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install it_deepface  
```

## Usage

```python
from it_deepface import FaceDuplicationDetection as FDD


# select two images path
path_img1, path_img2 = "./datasets/img_6.png", "./datasets/img_8.png"


# execute verification
fdd_response = FDD.verify_images(path_img1, path_img2)


# returns '{'verified': True, 'path_img': ['./datasets/img_6.png', './datasets/img_8.png'], 'threshold': 0.68}'
print(fdd_response)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)