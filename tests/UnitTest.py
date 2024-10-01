import unittest

def your_function(num1 , num2):
    return num1 + num2

class Functions_Test_Cases(unittest.TestCase):
    def test_case_1(self):
        # Test a specific case of your function
        result = your_function(2, 3)
        self.assertEqual(result, 5)  # Assuming the function adds two numbers
    
    def test_case_2(self):
        result = your_function(0, 0)
        self.assertEqual(result, 0)

if __name__ == '__main__':
    unittest.main()