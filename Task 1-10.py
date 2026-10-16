"""
=============================================================================
NAVTTC Artificial Intelligence (Machine Learning & Deep Learning) Tasks Sheet
Tasks 1 to 10
File: Task 1-10.py
=============================================================================
"""

import sys
import os
import cmath

def safe_input(prompt: str, default: str) -> str:
    """Helper function to allow user input if interactive, or use default if non-interactive."""
    try:
        if sys.stdin and sys.stdin.isatty():
            val = input(f"{prompt} (default '{default}'): ").strip()
            return val if val else default
        return default
    except (EOFError, OSError):
        return default


# =============================================================================
# TASK 1
# =============================================================================
"""
TASK 1: Installation
Statement:
Download and install Anaconda3
Install PyTorch
Install TensorFlow 2.0
Install VSCode
Install PyCharm
Week: 1
"""
print("=" * 70)
print("TASK 1: Software & Environment Installation Verification")
print("=" * 70)

def task_1_installation_check():
    print("Verifying installed AI/ML environment and libraries:\n")
    print(f"Python executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")

    # Check PyTorch
    try:
        import torch
        print(f"PyTorch Version   : {torch.__version__} (Installed)")
    except ImportError:
        print("PyTorch           : Not installed. Run: pip install torch torchvision")

    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__} (Installed)")
    except ImportError:
        print("TensorFlow        : Not installed. Run: pip install tensorflow")

    # Tools installation guide:
    print("\n[Installation Reference Guide]:")
    print("1. Anaconda3 : Download installer from https://www.anaconda.com/download")
    print("2. VSCode    : Download installer from https://code.visualstudio.com/")
    print("3. PyCharm   : Download Community Edition from https://www.jetbrains.com/pycharm/")
    print("4. PyTorch   : Command -> conda install pytorch torchvision -c pytorch")
    print("5. TensorFlow: Command -> pip install tensorflow")

task_1_installation_check()


# =============================================================================
# TASK 2
# =============================================================================
"""
TASK 2: Linux Commands
Statement:
Practice these commands:
pwd, cd, ls, cat, sudo, man, redirection, mkdir, rm, rmdir, cp, mv, file,
reading, cat, more, less, head, alias, shutdown, restart, touch, nano, bash,
sh, chmod, ps, kill, dpkg
Week: 1
"""
print("\n" + "=" * 70)
print("TASK 2: Linux Commands Simulation & Python Equivalence")
print("=" * 70)

def task_2_linux_commands():
    import shutil
    import subprocess
    from pathlib import Path

    print("Linux Command Reference & Python Equivalents:\n")
    commands_guide = [
        ("pwd", "Print Working Directory", "os.getcwd()", os.getcwd()),
        ("ls", "List Directory Contents", "os.listdir('.')", str(os.listdir('.'))),
        ("mkdir", "Make Directory", "os.makedirs('demo_dir', exist_ok=True)", "Creates a directory"),
        ("touch", "Create empty file / update timestamp", "Path('demo.txt').touch()", "Touches file"),
        ("cat", "Concatenate & display file content", "open('file').read()", "Reads full text"),
        ("head", "Display first N lines of file", "file.readlines()[:10]", "Shows head lines"),
        ("cp", "Copy file or folder", "shutil.copy(src, dst)", "Copies file"),
        ("mv", "Move or rename file", "shutil.move(src, dst)", "Moves/renames file"),
        ("rm", "Remove file", "os.remove(path)", "Deletes file"),
        ("rmdir", "Remove empty directory", "os.rmdir(path)", "Deletes empty dir"),
        ("chmod", "Change file permissions", "os.chmod(path, mode)", "Alters permissions"),
        ("ps", "Show running processes", "subprocess.run(['ps', 'aux'])", "Lists active processes"),
        ("kill", "Terminate process by PID", "os.kill(pid, signal.SIGTERM)", "Terminates process"),
        ("man", "Manual pages", "subprocess.run(['man', cmd])", "Displays command help"),
        ("sudo", "Execute command with superuser privileges", "sudo <command>", "Superuser privilege"),
        ("dpkg", "Debian package manager", "dpkg -i package.deb", "Manages .deb packages")
    ]

    for cmd, purpose, py_equiv, sample in commands_guide[:8]:
        print(f"Command   : {cmd}")
        print(f"Purpose   : {purpose}")
        print(f"Python    : {py_equiv}")
        print(f"Example/Value: {sample}")
        print("-" * 50)

task_2_linux_commands()


# =============================================================================
# TASK 3
# =============================================================================
"""
TASK 3: Python (Add Two Numbers)
"""
print("\n" + "=" * 70)
print("TASK 3: Python - Add Two Predefined Numbers")
print("=" * 70)

# Solution:
num1 = 1.5
num2 = 6.3

# Add two numbers
sum_result = num1 + num2

# Display the sum using format()
print('The sum of {0} and {1} is {2}'.format(num1, num2, sum_result))


# =============================================================================
# TASK 4
# =============================================================================
"""
TASK 4: Python (Add Two Numbers from User Input)
"""
print("\n" + "=" * 70)
print("TASK 4: Python - Add Two Numbers from User Input")
print("=" * 70)

# Solution:
input_val1 = safe_input("Enter first number", "12.4")
input_val2 = safe_input("Enter second number", "7.6")

num1 = float(input_val1)
num2 = float(input_val2)

# Add two numbers
total_sum = num1 + num2

# Display the sum
print('The sum of {0} and {1} is {2}'.format(num1, num2, total_sum))


# =============================================================================
# TASK 5
# =============================================================================
"""
TASK 5: Python (Calculate Square Root)
"""
print("\n" + "=" * 70)
print("TASK 5: Python - Calculate the Square Root")
print("=" * 70)

# Solution:
num_input = safe_input("Enter a number to find its square root", "8")
num = float(num_input)

# Exponentiation by 0.5 computes square root
num_sqrt = num ** 0.5
print('The square root of %0.3f is %0.3f' % (num, num_sqrt))


# =============================================================================
# TASK 6
# =============================================================================
"""
TASK 6: Python (Square Root of Real or Complex Numbers)
"""
print("\n" + "=" * 70)
print("TASK 6: Python - Square Root of Real or Complex Numbers")
print("=" * 70)

# Solution:
complex_input_str = safe_input("Enter a complex or real number", "1+2j")
try:
    num_complex = complex(complex_input_str)
except ValueError:
    num_complex = 1 + 2j

# Compute complex square root
num_sqrt_complex = cmath.sqrt(num_complex)
print('The square root of {0} is {1:0.3f}+{2:0.3f}j'.format(
    num_complex, num_sqrt_complex.real, num_sqrt_complex.imag
))


# =============================================================================
# TASK 7
# =============================================================================
"""
TASK 7: Python (Convert Temperature in Celsius to Fahrenheit)
fahrenheit = (celsius * 1.8) + 32
"""
print("\n" + "=" * 70)
print("TASK 7: Python - Convert Celsius to Fahrenheit")
print("=" * 70)

# Solution:
celsius_input = safe_input("Enter temperature in Celsius", "37.5")
celsius = float(celsius_input)

# Temperature conversion formula: F = (C * 1.8) + 32
fahrenheit = (celsius * 1.8) + 32
print('%0.1f degree Celsius is equal to %0.1f degree Fahrenheit' % (celsius, fahrenheit))


# =============================================================================
# TASK 8
# =============================================================================
"""
TASK 8: Python (Find Area of Triangle using Heron's Formula)
Statement:
"""
print("\n" + "=" * 70)
print("TASK 8: Python - Area of Triangle (Heron's Formula)")
print("=" * 70)

# Solution:
side_a = float(safe_input("Enter side a", "5"))
side_b = float(safe_input("Enter side b", "6"))
side_c = float(safe_input("Enter side c", "7"))

# Semi-perimeter calculation
s = (side_a + side_b + side_c) / 2.0

# Heron's formula for area
area = (s * (s - side_a) * (s - side_b) * (s - side_c)) ** 0.5
print('The area of the triangle with sides %0.1f, %0.1f, %0.1f is %0.2f' % (side_a, side_b, side_c, area))


# =============================================================================
# TASK 9
# =============================================================================
"""
TASK 9: Python (Solve Quadratic Equation ax^2 + bx + c = 0)
Statement:
Solve the quadratic equation ax**2 + bx + c = 0
"""
print("\n" + "=" * 70)
print("TASK 9: Python - Solve Quadratic Equation")
print("=" * 70)

# Solution:
qa = float(safe_input("Enter coefficient a", "1"))
qb = float(safe_input("Enter coefficient b", "5"))
qc = float(safe_input("Enter coefficient c", "6"))

# Calculate the discriminant
d = (qb ** 2) - (4 * qa * qc)

# Find two solutions (handles real and complex roots)
sol1 = (-qb - cmath.sqrt(d)) / (2 * qa)
sol2 = (-qb + cmath.sqrt(d)) / (2 * qa)

print(f"Quadratic Equation: ({qa})x^2 + ({qb})x + ({qc}) = 0")
print(f"Discriminant (b^2 - 4ac): {d}")
print('The solutions are {0} and {1}'.format(sol1, sol2))


# =============================================================================
# TASK 10
# =============================================================================
"""
TASK 10: Python (Convert Kilometers to Miles)
"""
print("\n" + "=" * 70)
print("TASK 10: Python - Convert Kilometers to Miles")
print("=" * 70)

# Solution:
km_input = safe_input("Enter value in kilometers", "100")
kilometers = float(km_input)

# Conversion factor: 1 kilometer = 0.621371 miles
conv_fac = 0.621371

# Calculate miles
miles = kilometers * conv_fac
print('%0.2f kilometers is equal to %0.2f miles' % (kilometers, miles))
print("=" * 70)

