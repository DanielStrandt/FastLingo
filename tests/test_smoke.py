import unittest
import numpy as np
from fastlingo.model import encode

class FastLingoSmokeTest(unittest.TestCase):
    def test_encoding_is_deterministic(self):
        kwargs=dict(dimensions=768, sparsity=4, n_values=(1,2,3,4), seed=20261108)
        self.assertTrue(np.array_equal(encode("Hello world", **kwargs), encode("Hello world", **kwargs)))

    def test_encoding_shape_and_dtype(self):
        value=encode("Hello world", dimensions=768, sparsity=4, n_values=(1,2,3,4))
        self.assertEqual(value.shape, (8,768))
        self.assertEqual(value.dtype, np.int8)

if __name__ == "__main__":
    unittest.main()
