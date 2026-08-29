from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from src.inference import predict_image, prepare_image


class FakeModel:
    def __init__(self, score: float):
        self.score = score

    def predict(self, batch, verbose=0):
        self.batch_shape = batch.shape
        return np.array([[self.score]], dtype=np.float32)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.image_path = Path(self.temp_dir.name) / "xray.png"
        Image.new("RGB", (30, 40), color=(20, 40, 60)).save(self.image_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_prepare_image_matches_model_contract(self):
        batch = prepare_image(self.image_path)
        self.assertEqual(batch.shape, (1, 224, 224, 1))
        self.assertEqual(batch.dtype, np.float32)

    def test_prediction_uses_configured_threshold(self):
        model = FakeModel(0.61)
        result = predict_image(self.image_path, model=model, threshold=0.60)
        self.assertEqual(result.label, "PNEUMONIA")
        self.assertEqual(model.batch_shape, (1, 224, 224, 1))

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_image(self.image_path, model=FakeModel(0.5), threshold=1.0)


if __name__ == "__main__":
    unittest.main()
