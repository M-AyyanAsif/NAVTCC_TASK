
# =============================================================================
# TASK 11
# =============================================================================
"""
TASK 11: Python (Simple 'if' Statement)

"""
print("=" * 70)
print("TASK 11: Python - Simple 'if' Statement Demonstration")
print("=" * 70)

# Solution:
i = 10
if (i > 15):
    print("10 is greater than 15")
print("I am Not in if (Condition i > 15 evaluated to False for i = 10)")


# =============================================================================
# TASK 12
# =============================================================================
"""
TASK 12: Python ('if-else' Statement)

"""
print("\n" + "=" * 70)
print("TASK 12: Python - 'if-else' Conditional Statement")
print("=" * 70)

# Solution:
i = 20
if (i < 15):
    print("i is smaller than 15")
    print("i'm in if Block")
else:
    print("i is greater than 15")
    print("i'm in else Block")
print("i'm not in if and not in else Block")


# =============================================================================
# TASK 13
# =============================================================================
"""
TASK 13: Python (Nested 'if' Statement)

"""
print("\n" + "=" * 70)
print("TASK 13: Python - Nested 'if' Statements")
print("=" * 70)

# Solution:
i = 10
if (i == 10):
    # First if statement
    if (i < 15):
        print("i is smaller than 15")
    # Nested if statement
    if (i < 12):
        print("i is smaller than 12 too")
    else:
        print("i is greater than 15")


# =============================================================================
# TASK 14
# =============================================================================
"""
TASK 14: Python ('if-elif-else' Ladder)

"""
print("\n" + "=" * 70)
print("TASK 14: Python - 'if-elif-else' Ladder")
print("=" * 70)

# Solution:
i = 20
if (i == 10):
    print("i is 10")
elif (i == 15):
    print("i is 15")
elif (i == 20):
    print("i is 20")
else:
    print("i is not present")


# =============================================================================
# TASK 15
# =============================================================================
"""
TASK 15: Python (Exercise on 'for' Loops)
Reference Link: https://www.geeksforgeeks.org/python-for-loops/

"""
print("\n" + "=" * 70)
print("TASK 15: Python - Exercises on 'for' Loops")
print("=" * 70)

# 1. Iterating over list
fruits = ["Apple", "Banana", "Cherry"]
print("1. Iterating through list:")
for fruit in fruits:
    print(f"   Item: {fruit}")

# 2. Using range(start, stop, step)
print("\n2. Using range(2, 11, 2) [Even numbers up to 10]:")
for n in range(2, 11, 2):
    print(f"   {n}", end=" ")
print()

# 3. Nested for loops (Multiplication table snippet)
print("\n3. Nested for loops (2x3 grid):")
for r in range(1, 3):
    for c in range(1, 4):
        print(f"   Cell({r},{c}) = {r*c}", end=" | ")
    print()

# 4. for-else block (executes only if loop completes without break)
print("\n4. for-else construct:")
search_target = 99
numbers = [10, 20, 30, 40]
for num in numbers:
    if num == search_target:
        print(f"   Found {search_target}!")
        break
else:
    print(f"   {search_target} was not found in the list (else block executed).")


# =============================================================================
# TASK 16
# =============================================================================
"""
TASK 16: Python (Exercise on 'while' Loops)
Reference Link: https://www.geeksforgeeks.org/python-while-loops/

"""
print("\n" + "=" * 70)
print("TASK 16: Python - Exercises on 'while' Loops")
print("=" * 70)

# 1. Counter-controlled while loop
print("1. Countdown using while loop:")
count = 5
while count > 0:
    print(f"   Countdown: {count}")
    count -= 1
print("   Blast off!")

# 2. Summation of first N numbers
n = 10
total_sum = 0
curr = 1
while curr <= n:
    total_sum += curr
    curr += 1
print(f"\n2. Sum of first {n} natural numbers is: {total_sum}")

# 3. while-else loop
print("\n3. while-else construct:")
x = 1
while x <= 3:
    print(f"   Loop counter x = {x}")
    x += 1
else:
    print("   while loop condition became False; else block triggered.")


# =============================================================================
# TASK 17
# =============================================================================
"""
TASK 17: Python (Exercise on 'break' Statement)
Reference Link: https://www.geeksforgeeks.org/python-break-statement/

"""
print("\n" + "=" * 70)
print("TASK 17: Python - Exercises on 'break' Statement")
print("=" * 70)

# 1. Break on target element
data = ["AI", "Machine Learning", "Deep Learning", "Robotics", "NLP"]
target = "Deep Learning"
print(f"Searching for '{target}' in list:")
for item in data:
    print(f"   Inspecting: {item}")
    if item == target:
        print(f"   --> Found target '{target}'! Breaking loop now.")
        break

# 2. Infinite loop break
print("\nBreaking out of a while True loop with a threshold:")
counter = 1
while True:
    print(f"   Iteration {counter}")
    if counter >= 3:
        print("   Counter reached 3 -> executing 'break'.")
        break
    counter += 1


# =============================================================================
# TASK 18
# =============================================================================
"""
TASK 18: Python (Exercise on 'continue' Statement)
Reference Link: https://www.geeksforgeeks.org/python-continue-statement/

"""
print("\n" + "=" * 70)
print("TASK 18: Python - Exercises on 'continue' Statement")
print("=" * 70)

# 1. Filter out odd numbers using continue
print("1. Printing only even numbers between 1 and 10:")
for num in range(1, 11):
    if num % 2 != 0:
        continue  # Skip odd numbers
    print(f"   Even number: {num}")

# 2. Skip unwanted characters in string
raw_text = "N-A-V-T-T-C"
clean_text = ""
for char in raw_text:
    if char == "-":
        continue
    clean_text += char
print(f"\n2. Cleaned text (skipping hyphens): '{clean_text}'")


# =============================================================================
# TASK 19
# =============================================================================
"""
TASK 19: Python (Exercise on Various Looping Techniques)
Reference Link: https://www.geeksforgeeks.org/looping-techniques-python/

"""
print("\n" + "=" * 70)
print("TASK 19: Python - Looping Techniques")
print("=" * 70)

courses = ["Data Science", "Machine Learning", "Deep Learning", "Azure AI"]
codes = ["DS101", "ML201", "DL301", "AZ401"]

# 1. enumerate()
print("1. Using enumerate():")
for index, course in enumerate(courses, start=1):
    print(f"   {index}. Course Name: {course}")

# 2. zip()
print("\n2. Using zip() to pair courses with codes:")
for code, course in zip(codes, courses):
    print(f"   Code: {code} -> Course: {course}")

# 3. reversed()
print("\n3. Using reversed() without mutating original list:")
for item in reversed(courses):
    print(f"   Reverse item: {item}")

# 4. sorted()
scores = [88, 95, 72, 64, 91]
print("\n4. Using sorted() on scores:")
for score in sorted(scores):
    print(f"   Score: {score}", end=" | ")
print()

# 5. dict.items()
course_duration_weeks = {"Python": 4, "ML": 6, "DL": 4, "Azure": 2}
print("\n5. Using dict.items():")
for subject, weeks in course_duration_weeks.items():
    print(f"   Subject: {subject:<10} Duration: {weeks} weeks")


# =============================================================================
# TASK 20
# =============================================================================
"""
TASK 20: Python (Exercise on User Defined Functions)
Reference Link: https://www.geeksforgeeks.org/functions-in-python/

"""
print("\n" + "=" * 70)
print("TASK 20: Python - User Defined Functions")
print("=" * 70)

# 1. Simple function with calculation and return
def calculate_grade(score: float) -> str:
    """Determine grade based on numeric score."""
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    else:
        return "C"

print("1. calculate_grade(85.5) ->", calculate_grade(85.5))

# 2. Function with default parameters
def greet_student(name: str, institution: str = "NAVTTC Center") -> str:
    """Greet student with default institution."""
    return f"Welcome, {name} to {institution}!"

print("2. greet_student('Ali') ->", greet_student("Ali"))
print("   greet_student('Sara', 'FAST NUCES') ->", greet_student("Sara", "FAST NUCES"))

# 3. Arbitrary Positional Arguments (*args)
def compute_average(*scores: float) -> float:
    """Calculate mean of any number of given numbers."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

print("3. compute_average(80, 90, 100, 70) ->", compute_average(80, 90, 100, 70))

# 4. Arbitrary Keyword Arguments (**kwargs)
def build_student_profile(roll_no: int, **details) -> dict:
    """Package student details dynamically into a profile dictionary."""
    profile = {"roll_no": roll_no}
    profile.update(details)
    return profile

student = build_student_profile(
    101,
    name="Muhammad Ayyan",
    course="Artificial Intelligence",
    city="Lahore",
    enrolled=True
)
print("4. build_student_profile(...) ->")
for k, v in student.items():
    print(f"   {k}: {v}")
print("=" * 70)

