import unittest
from utils import hello

class TestUtils(unittest.TestCase):
    def test_hello(self):
        self.assertEqual(hello(), "Hello")

if __name__ == '__main__':
    unittest.main()
