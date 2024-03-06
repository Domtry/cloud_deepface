# common dependencies
import os
import logging
import warnings
from typing import Any, Dict, Union, Optional

# 3rd party dependencies
import pandas as pd
import tensorflow as tf

# package dependencies
from cloud_deepface.commons import package_utils, folder_utils
from cloud_deepface.commons.logger import Logger
from cloud_deepface.drivers import config
from cloud_deepface.setting import DetectorBackend, DistanceMetric, ModelName
from cloud_deepface.modules import (
    modeling,
    recognition,
    verification
)

logger = Logger(module="DeepFace")

# current package version of deepface
__version__ = package_utils.find_package_version()

# -----------------------------------
# configurations for dependencies

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf_version = package_utils.get_tf_major_version()
if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)
# -----------------------------------

# create required folders if necessary to store model weights
folder_utils.initialize_folder()


def build_model(model_name: str) -> Any:
    """
    This function builds a deepface model
    Args:
        model_name (string): face recognition or facial attribute model
            VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
            Age, Gender, Emotion, Race for facial attributes
    Returns:
        built_model
    """
    return modeling.build_model(model_name=model_name)


def find_from_cloud(
    img_url: str,
    bucket_name: str,
    driver: Optional[str] = None,
    synchronization: bool = False,
    model_name: ModelName = ModelName.FACENET,
    detector_backend: DetectorBackend = DetectorBackend.OPEN_CV,
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN_L2,
    align: bool = True,
    silent: bool = False,
    enforce_detection: bool = False,
    expand_percentage: int = 0,
    threshold: Optional[float] = None,
    normalization: str = "base",
) -> Union[Dict[str, Any], Exception]:

    urls: list[str] = []
    
    __model_name = model_name.value
    __distance_metric = distance_metric.value
    __detector_backend = detector_backend.value
    
    filename = f"representations_{__model_name}.pkl".lower()
    
    client_driver = config.load_driver(driver)
    
    if synchronization :
        urls = client_driver.load_data(bucket_name)
        
    else:
        if not (cloud_data := client_driver.get_representation(filename)) :
            raise FileNotFoundError(f"{filename} not found. Please active synchronization")
        else:
            urls = eval(cloud_data)
            
    result = recognition.find_from_cloud(
        urls=urls,
        img_url=img_url,
        driver=client_driver,
        filename=filename,
        model_name=__model_name,
        synchronization=synchronization,
        distance_metric=__distance_metric,
        enforce_detection=enforce_detection,
        detector_backend=__detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        threshold=threshold,
        normalization=normalization,
        silent=silent,
    )

    threshold = verification.find_threshold(
        model_name=__detector_backend,
        distance_metric=__distance_metric
    )

    data_frame: pd.DataFrame = result[0]

    return {
        "verified": len(data_frame.values) > 0,
        "threshold": threshold,
        "path_img": [item[0] for item in data_frame.values.tolist()],
    }
