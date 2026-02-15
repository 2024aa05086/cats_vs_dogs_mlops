import numpy as np
import tensorflow as tf
from src.api.model_loader import predict_image


class DummyModel:
    def predict(self, x, verbose=0):
        # Return constant probability for dog
        return np.array([[0.8]], dtype=np.float32)


def test_predict_image_returns_expected():
    model = DummyModel()
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    pred, prob = predict_image(model, img)

    assert pred == 1
    assert 0.0 <= prob <= 1.0
