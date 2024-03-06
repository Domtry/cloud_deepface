# standard module
import pytest

# local module
from src.cloud_deepface import DeepFace


class TestCloudDeepFace:

    def test_find_from_cloud_by_default_config(self):
        got_url = "https://www.shutterstock.com/image-photo/image-african-man-foot-wearworkshop-600nw-1839501016.jpg"
        
        response = DeepFace.find_from_cloud(
            img_url=got_url,
            synchronization=False,
            bucket_name="storages",
            driver="minio",
            bin_path="./datasets")

        assert response["verified"] is True
        assert response["path_img"] is not None
        assert len(response["path_img"]) == 1
        
        
        
    def test_find_from_cloud_activate_synchronization(self):
        got_url = "https://www.shutterstock.com/image-photo/image-african-man-foot-wearworkshop-600nw-1839501016.jpg"
        
        response = DeepFace.find_from_cloud(
            img_url=got_url,
            synchronization=True,
            bucket_name="storages",
            driver="minio")

        assert response["verified"] is True
        assert response["path_img"] is not None
        assert len(response["path_img"]) == 1
        
    
    def test_find_from_cloud_invalid_bin_path(self):
        
        with pytest.raises(FileNotFoundError) as ex_info:
            
            got_url = "https://www.shutterstock.com/image-photo/image-african-man-foot-wearworkshop-600nw-1839501016.jpg"
            
            DeepFace.find_from_cloud(
                img_url=got_url,
                synchronization=True,
                bucket_name="storages",
                driver="minio")

            assert str(ex_info.value) == "No such file or directory: './datsets/representations_deepface.pkl'"
