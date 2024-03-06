from enum import Enum


class ModelName(Enum):
    VGG_FACE = "VGG-Face"
    FACENET = "Facenet"
    FACENET_512 = "Facenet512"
    OPEN_FACE = "OpenFace"
    DEEP_FACE = "DeepFace"
    DEEP_ID = "DeepID"
    ARC_FACE = "ArcFace"
    D_LIB = "Dlib"
    S_FACE = "SFace"
    
    
class DetectorBackend(Enum):
    OPEN_CV = "opencv"
    SSD = "ssd"
    DLIB = "dlib"
    MTCNN = "mtcnn"
    RETINA_FACE = "retinaface"
    MEDIA_PIPE = "mediapipe"
    YOLOV_8 = "yolov8"
    YUNET = "yunet"
    FASTM_TCNN = "fastmtcnn"
    

class DistanceMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    EUCLIDEAN_L2 = "euclidean_l2"