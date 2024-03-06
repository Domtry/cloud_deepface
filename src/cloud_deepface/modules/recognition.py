# built-in dependencies
from typing import (
    List, 
    Union, 
    Optional, 
    Any)
import time

# 3rd party dependencies
import numpy as np
import pandas as pd
from tqdm import tqdm

# project dependencies
from cloud_deepface.commons.logger import Logger
from cloud_deepface.drivers.DriverAbstract import DriverAbstract as Driver
from cloud_deepface.modules import representation, detection, modeling, verification
from cloud_deepface.models.FacialRecognition import FacialRecognition

logger = Logger(module="it_deepface/modules/recognition.py")


def find_from_cloud(
    img_url: Union[str, np.ndarray],
    urls: list[str],
    driver: Driver,
    filename: str,
    synchronization:bool,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    threshold: Optional[float] = None,
    normalization: str = "base",
    silent: bool = False,
) -> List[pd.DataFrame]:
    
    tic = time.time()
    representations:list[Any] = []
    model: FacialRecognition = modeling.build_model(model_name)
    target_size = model.input_shape

    # ---------------------------------------
    
    df_cols = [
        "identity",
        f"{model_name}_representation",
        "target_x",
        "target_y",
        "target_w",
        "target_h",
    ]

    employees = urls

    if len(employees) == 0:
        raise ValueError(
            f"There is no image in cloud folder!"
            "Validate .jpg, .jpeg or .png files exist in this path.",
        )

    if synchronization :
        representations = __find_bulk_embeddings(
            employees=employees,
            model_name=model_name,
            target_size=target_size,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            normalization=normalization,
            silent=silent,
        )
        
        driver.upload_representation(filename, f"{representations}".encode("utf-8"))
    else:
        
        bytes_data = driver.get_representation(filename)
        representations = eval(bytes_data)

    df = pd.DataFrame(
        representations,
        columns=df_cols,
    )

    # img path might have more than once face
    source_objs = detection.extract_faces(
        img_path=img_url,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
    )

    resp_obj = []

    for source_obj in source_objs:
        source_img = source_obj["face"]
        source_region = source_obj["facial_area"]
        target_embedding_obj = representation.represent(
            img_path=source_img,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,
        )

        target_representation = target_embedding_obj[0]["embedding"]

        result_df = df.copy()  # df will be filtered in each img
        result_df["source_x"] = source_region["x"]
        result_df["source_y"] = source_region["y"]
        result_df["source_w"] = source_region["w"]
        result_df["source_h"] = source_region["h"]

        distances = []
        for _, instance in df.iterrows():
            source_representation = instance[f"{model_name}_representation"]

            target_dims = len(list(target_representation))
            source_dims = len(list(source_representation))
            if target_dims != source_dims:
                raise ValueError(
                    "Source and target embeddings must have same dimensions but "
                    + f"{target_dims}:{source_dims}. Model structure may change"
                    + " after pickle created. Delete the {file_name} and re-run."
                )

            if distance_metric == "cosine":
                distance = verification.find_cosine_distance(
                    source_representation, target_representation
                )
            elif distance_metric == "euclidean":
                distance = verification.find_euclidean_distance(
                    source_representation, target_representation
                )
            elif distance_metric == "euclidean_l2":
                distance = verification.find_euclidean_distance(
                    verification.l2_normalize(source_representation),
                    verification.l2_normalize(target_representation),
                )
            else:
                raise ValueError(f"invalid distance metric passes - {distance_metric}")

            distances.append(distance)

            # ---------------------------
        target_threshold = threshold or verification.find_threshold(model_name, distance_metric)

        result_df["threshold"] = target_threshold
        result_df["distance"] = distances

        result_df = result_df.drop(columns=[f"{model_name}_representation"])
        # pylint: disable=unsubscriptable-object
        result_df = result_df[result_df["distance"] <= target_threshold]
        result_df = result_df.sort_values(by=["distance"], ascending=True).reset_index(drop=True)

        resp_obj.append(result_df)

    # -----------------------------------

    toc = time.time()

    if not silent:
        logger.info(f"find function lasts {toc - tic} seconds")
    
    return resp_obj


def __find_bulk_embeddings(
    employees: List[str],
    model_name: str = "VGG-Face",
    target_size: tuple = (224, 224),
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    silent: bool = False,
):
    """
    Find embeddings of a list of images

    Args:
        employees (list): list of exact image paths

        model_name (str): facial recognition model name

        target_size (tuple): expected input shape of facial recognition model

        detector_backend (str): face detector model name

        enforce_detection (bool): set this to False if you
            want to proceed when you cannot detect any face

        align (bool): enable or disable alignment of image
            before feeding to facial recognition model

        expand_percentage (int): expand detected facial area with a
            percentage (default is 0).

        normalization (bool): normalization technique

        silent (bool): enable or disable informative logging
    Returns:
        representations (list): pivot list of embeddings with
            image name and detected face area's coordinates
    """
    representations = []
    for employee in tqdm(
        employees,
        desc="Finding representations",
        disable=silent,
    ):
        img_objs = detection.extract_faces(
            img_path=employee,
            target_size=target_size,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
        )

        for img_obj in img_objs:
            img_content = img_obj["face"]
            img_region = img_obj["facial_area"]
            embedding_obj = representation.represent(
                img_path=img_content,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            img_representation = embedding_obj[0]["embedding"]

            instance = []
            instance.append(employee)
            instance.append(img_representation)
            instance.append(img_region["x"])
            instance.append(img_region["y"])
            instance.append(img_region["w"])
            instance.append(img_region["h"])
            representations.append(instance)
    return representations
