import unittest

from src.universe.system import file_count, get_files


class TestFile(unittest.TestCase):
    def test_file_count(self):
        self.assertEqual(file_count('test_data'), 0)
        self.assertEqual(file_count('test_data', recursive=True), 3)
        self.assertEqual(file_count('test_data', extension='pkl'), 0)
        self.assertEqual(file_count('test_data', extension='pkl', recursive=True), 2)

    def test_get_files(self):
        self.assertEqual(get_files('test_data'), [])
        self.assertEqual(get_files('test_data', recursive=True), [
            'test_data\\data\\cvat\\annotations.xml',
            'test_data\\data\\cvat\\boxes.pkl',
            'test_data\\data\\cvat\\polylines.pkl'
        ])
        self.assertEqual(get_files('test_data', extension='pkl'), [])
        self.assertEqual(get_files('test_data', extension='pkl', recursive=True), [
            'test_data\\data\\cvat\\boxes.pkl',
            'test_data\\data\\cvat\\polylines.pkl'
        ])
