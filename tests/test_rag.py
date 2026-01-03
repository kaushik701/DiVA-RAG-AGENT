import unittest
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import load_diabetes_data

class TestRAGComponents(unittest.TestCase):
    
    def test_config_paths(self):
        """Test if data file path is configured correctly"""
        self.assertTrue(os.path.exists(config.DATA_DIR), "Data directory does not exist")
        # We check if the file path string is constructed, not necessarily if file exists for this unit test
        self.assertTrue(config.JSON_FILE_PATH.endswith(".json"))

    def test_data_loader(self):
        """Test if data loader returns a list"""
        if os.path.exists(config.JSON_FILE_PATH):
            docs = load_diabetes_data(config.JSON_FILE_PATH)
            self.assertIsInstance(docs, list)
            self.assertGreater(len(docs), 0)

if __name__ == '__main__':
    unittest.main()