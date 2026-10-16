
import numpy as np

# =============================================================================
# TASK 21
# =============================================================================
"""
TASK 21: Python (Exercise on List Data Type)
Reference Link: https://www.programiz.com/python-programming/list

"""
print("=" * 70)
print("TASK 21: Python - List Data Type Exercises")
print("=" * 70)

# 1. Creation and Indexing
numbers = [10, 20, 30, 40, 50]
print(f"Original list: {numbers}")
print(f"First element (index 0) : {numbers[0]}")
print(f"Last element (index -1) : {numbers[-1]}")

# 2. Slicing
print(f"Slice [1:4]             : {numbers[1:4]}")
print(f"Slice with step [::2]   : {numbers[::2]}")

# 3. List Methods
numbers.append(60)
print(f"After append(60)        : {numbers}")
numbers.extend([70, 80])
print(f"After extend([70, 80])  : {numbers}")
numbers.insert(2, 25)
print(f"After insert(2, 25)     : {numbers}")
removed_val = numbers.pop()
print(f"Popped last item ({removed_val}) : {numbers}")
numbers.remove(25)
print(f"After remove(25)        : {numbers}")

# 4. List Comprehension
squares = [x ** 2 for x in range(1, 6)]
print(f"List comprehension (squares of 1-5): {squares}")


# =============================================================================
# TASK 22
# =============================================================================
"""
TASK 22: Python (Exercise on Tuple Data Type)
Reference Link: https://www.programiz.com/python-programming/tuple

Week: 1
"""
print("\n" + "=" * 70)
print("TASK 22: Python - Tuple Data Type Exercises")
print("=" * 70)

# 1. Creation
single_item_tuple = ("AI",)  # note trailing comma
tech_tuple = ("Python", "Machine Learning", "Deep Learning", "Python")
print(f"Single-element tuple    : {single_item_tuple} (type: {type(single_item_tuple).__name__})")
print(f"Multi-element tuple     : {tech_tuple}")

# 2. Immutability
try:
    tech_tuple[0] = "Java"  # type: ignore
except TypeError as e:
    print(f"Immutability verified   : TypeError caught -> {e}")

# 3. Tuple Methods
print(f"Count of 'Python'       : {tech_tuple.count('Python')}")
print(f"Index of 'Deep Learning': {tech_tuple.index('Deep Learning')}")

# 4. Tuple Unpacking
point_3d = (12, 25, 50)
x, y, z = point_3d
print(f"Unpacked point_3d (x={x}, y={y}, z={z})")


# =============================================================================
# TASK 23
# =============================================================================
"""
TASK 23: Python (Exercise on String Data Type)
Reference Link: https://www.programiz.com/python-programming/string

"""
print("\n" + "=" * 70)
print("TASK 23: Python - String Data Type Exercises")
print("=" * 70)

message = "   National Vocational and Technical Training Commission   "

# 1. Slicing and Stripping
clean_msg = message.strip()
print(f"Raw string with spaces : '{message}'")
print(f"Stripped string        : '{clean_msg}'")
print(f"First 8 characters     : '{clean_msg[:8]}'")
print(f"Reversed string        : '{clean_msg[::-1][:15]}...'")

# 2. String Methods
print(f"Uppercase              : '{clean_msg.upper()}'")
print(f"Lowercase              : '{clean_msg.lower()}'")
print(f"Replaced string        : '{clean_msg.replace('Commission', 'Pakistan')}'")

words = clean_msg.split()
print(f"Split into words       : {words}")
joined = "-".join(words)
print(f"Joined with '-'        : '{joined}'")

# 3. String formatting
name = "Ayyan"
domain = "Artificial Intelligence"
print(f"F-String formatting    : Student {name} is enrolled in {domain}.")


# =============================================================================
# TASK 24
# =============================================================================
"""
TASK 24: Python (Exercise on Set Data Type)
Reference Link: https://www.programiz.com/python-programming/set

"""
print("\n" + "=" * 70)
print("TASK 24: Python - Set Data Type Exercises")
print("=" * 70)

# 1. Set Creation and Uniqueness
raw_data = [1, 2, 2, 3, 4, 4, 4, 5]
unique_set = set(raw_data)
print(f"List with duplicates   : {raw_data}")
print(f"Set (duplicates removed): {unique_set}")

# 2. Add and Discard
unique_set.add(6)
unique_set.discard(2)
print(f"After adding 6, removing 2: {unique_set}")

# 3. Set Operations
set_a = {1, 2, 3, 4, 5}
set_b = {4, 5, 6, 7, 8}
print(f"Set A: {set_a}")
print(f"Set B: {set_b}")

print(f"Union (A | B)                  : {set_a | set_b}")
print(f"Intersection (A & B)           : {set_a & set_b}")
print(f"Difference (A - B)             : {set_a - set_b}")
print(f"Symmetric Difference (A ^ B)   : {set_a ^ set_b}")


# =============================================================================
# TASK 25
# =============================================================================
"""
TASK 25: Python (Exercise on Dictionary Data Type)
Reference Link: https://www.programiz.com/python-programming/dictionary

"""
print("\n" + "=" * 70)
print("TASK 25: Python - Dictionary Data Type Exercises")
print("=" * 70)

# 1. Creation and Access
course_info = {
    "title": "AI, ML & Deep Learning",
    "duration_months": 3,
    "practical_ratio": "80%",
    "theory_ratio": "20%"
}
print(f"Course Dictionary: {course_info}")
print(f"Title using ['title']    : {course_info['title']}")
print(f"Safe access with .get()  : {course_info.get('certification', 'NAVTTC Certified')}")

# 2. Update and Remove
course_info["instructor"] = "Senior AI Trainer"
popped_ratio = course_info.pop("theory_ratio")
print(f"After update & pop: {course_info}")

# 3. Iterating
print("Iterating through dictionary:")
for key, value in course_info.items():
    print(f"   {key:<18} : {value}")

# 4. Dictionary Comprehension
square_dict = {x: x**2 for x in range(1, 6)}
print(f"Dictionary comprehension (n: n^2): {square_dict}")


# =============================================================================
# TASK 26
# =============================================================================
"""
TASK 26: Python (Exercise on Exception Handling)
Reference Link: https://www.programiz.com/python-programming/exception-handling

"""
print("\n" + "=" * 70)
print("TASK 26: Python - Exception Handling Exercises")
print("=" * 70)

def divide_numbers(numerator: float, denominator: float):
    """Safely divide numbers with full try-except-else-finally handling."""
    print(f"\nAttempting division: {numerator} / {denominator}")
    try:
        result = numerator / denominator
    except ZeroDivisionError as err:
        print(f"   [Error Caught]: Cannot divide by zero! ({err})")
    except TypeError as err:
        print(f"   [Error Caught]: Invalid operand types! ({err})")
    else:
        print(f"   [Success]: Result is {result:.4f}")
    finally:
        print("   [Finally Block]: Cleanup / execution complete.")

divide_numbers(100, 4)
divide_numbers(100, 0)


# =============================================================================
# TASK 27
# =============================================================================
"""
TASK 27: Python (Exercise on User Defined Exception Handling)
Reference Link: https://www.programiz.com/python-programming/user-defined-exception

"""
print("\n" + "=" * 70)
print("TASK 27: Python - User Defined Exception Handling")
print("=" * 70)

# Custom Exception Class
class InvalidAgeError(Exception):
    """Exception raised when an applicant's age does not meet requirements."""
    def __init__(self, age: int, message: str = "Age must be between 18 and 35 to apply for NAVTTC program."):
        self.age = age
        self.message = message
        super().__init__(f"InvalidAgeError: {message} (Given Age: {age})")

def verify_candidate_eligibility(age: int):
    """Validate candidate age against NAVTTC admission criteria."""
    print(f"\nVerifying applicant with age = {age}...")
    try:
        if age < 18 or age > 35:
            raise InvalidAgeError(age)
        print(f"   Eligibility check passed for age {age}!")
    except InvalidAgeError as e:
        print(f"   [Handled Custom Exception]: {e}")

verify_candidate_eligibility(22)
verify_candidate_eligibility(15)
verify_candidate_eligibility(42)


# =============================================================================
# TASK 28
# =============================================================================
"""
TASK 28: NumPy (Exercise on NumPy Array Creation)
Reference Link: https://www.w3schools.com/python/numpy_creating_arrays.asp

"""
print("\n" + "=" * 70)
print("TASK 28: NumPy - Array Creation Exercises")
print("=" * 70)

# 1. Multi-dimensional Arrays
arr_0d = np.array(42)
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"0-D Array: {arr_0d} | ndim: {arr_0d.ndim}")
print(f"1-D Array: {arr_1d} | ndim: {arr_1d.ndim}")
print(f"2-D Array:\n{arr_2d}\n   ndim: {arr_2d.ndim}")
print(f"3-D Array:\n{arr_3d}\n   ndim: {arr_3d.ndim}")

# 2. Built-in creators
zeros_arr = np.zeros((2, 3))
ones_arr = np.ones((2, 2))
range_arr = np.arange(10, 25, 3)
linspace_arr = np.linspace(0, 1, 5)

print(f"\nnp.zeros((2, 3)):\n{zeros_arr}")
print(f"np.ones((2, 2)):\n{ones_arr}")
print(f"np.arange(10, 25, 3): {range_arr}")
print(f"np.linspace(0, 1, 5): {linspace_arr}")


# =============================================================================
# TASK 29
# =============================================================================
"""
TASK 29: NumPy (Exercise on NumPy Array Indexing)
Reference Link: https://www.w3schools.com/python/numpy_array_indexing.asp

"""
print("\n" + "=" * 70)
print("TASK 29: NumPy - Array Indexing Exercises")
print("=" * 70)

# 1. 1D Indexing
arr_1d = np.array([10, 20, 30, 40, 50])
print(f"1D Array: {arr_1d}")
print(f"Element at index 0 : {arr_1d[0]}")
print(f"Element at index 3 : {arr_1d[3]}")
print(f"Element at index -1: {arr_1d[-1]}")

# 2. 2D Indexing
arr_2d = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print(f"\n2D Array:\n{arr_2d}")
print(f"Element at row 0, col 1 [0, 1] : {arr_2d[0, 1]}")
print(f"Element at row 2, col 2 [2, 2] : {arr_2d[2, 2]}")
print(f"Last row, last column [-1, -1] : {arr_2d[-1, -1]}")


# =============================================================================
# TASK 30
# =============================================================================
"""
TASK 30: NumPy (Exercise on NumPy Array Slicing)
Reference Link: https://www.w3schools.com/python/numpy_array_slicing.asp

"""
print("\n" + "=" * 70)
print("TASK 30: NumPy - Array Slicing Exercises")
print("=" * 70)

# 1. 1D Slicing
arr_1d = np.array([10, 20, 30, 40, 50, 60, 70, 80])
print(f"Original 1D Array: {arr_1d}")
print(f"Slice from index 2 to 6 [2:6]     : {arr_1d[2:6]}")
print(f"Slice from index 3 to end [3:]     : {arr_1d[3:]}")
print(f"Slice from beginning to 4 [:4]     : {arr_1d[:4]}")
print(f"Slice with step 2 [1:7:2]          : {arr_1d[1:7:2]}")

# 2. 2D Slicing
arr_2d = np.array([
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12]
])
print(f"\nOriginal 2D Matrix (3x4):\n{arr_2d}")
print(f"Extract rows 0 to 1, column 2 arr[0:2, 2]:\n{arr_2d[0:2, 2]}")
print(f"Extract 2x2 top-right sub-matrix arr[0:2, 2:4]:\n{arr_2d[0:2, 2:4]}")
print(f"Extract all rows, alternating columns arr[:, ::2]:\n{arr_2d[:, ::2]}")
print("=" * 70)

