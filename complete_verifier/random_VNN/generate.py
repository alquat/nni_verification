'''
Fenerate vnnlib property files and random.csv for random  torch samples

'''

from typing import Tuple, List, Any

from math import pi

import numpy as np

def main():
    filename = "random_instances.csv"
    print(f"Generating {filename}...")
    with open(filename, 'w') as f:   
        f.write(f'{"random.onnx"},{"random.vnnlib"},{"116"}\n')

if __name__ == "__main__":
    main()
    