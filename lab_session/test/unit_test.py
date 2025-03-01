import unittest

import sys
import os

# Add the path to the ml_code module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_code.data_load import load_data

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        data = load_data("data/data.csv")
        self.assertisNotNone(data)
        self.assertEqual(len(data), 150)

if __name__ == '__main__':
    unittest.main()