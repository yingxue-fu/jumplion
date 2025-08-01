# builtin
import os
import re
# external
from tqdm import tqdm
import numpy as np

#
MSFormatsExt = {"mzml", "raw", "d"}  
MaxLibFrNum = 20

#
E = np.float64(0.000000001)
INF = np.float64(10000000.0)

#
C13delta = 1.003355
proton = 1.007825035
OH = 17.003288

#
fTypeB = 1 << 0      # 00000001 (binary) or 1 (decimal)
fTypeY = 1 << 1      # 00000010 (binary) or 2 (decimal)
char_of_fType = ['?', 'b', 'y']
fExclude = 1 << 6   

# 
name_of_loss = ["noloss", "H2O", "NH3", "CO", "unknown", "unknown"]
loss_none, loss_H2O, loss_NH3, loss_CO, loss_N, loss_other = np.arange(6)
Loss = np.array([0.0, 18.011113035, 17.026548, 27.994915, 0.0, 0.0], dtype=np.float32)


def progress_bar(iterable, show=True, intermittent=False):
    total = len(iterable)
    disable = not show
    if not intermittent:
        for item in tqdm(iterable, total=total, disable=disable, leave=True):
            yield item
    else:
        steps=5
        percent_points = [int(round(p)) for p in 
                          [i * 100 / (steps - 1) for i in range(1, steps)]]
        milestones = sorted({int(total * p / 100) for p in percent_points})

        with tqdm(total=total, disable=disable, leave=False) as pbar:
            for count, item in enumerate(iterable, 1):
                yield item
                if count in milestones:
                    pbar.n = count
                    pbar.refresh()


def fast_trim(string):
    return string.strip()

def get_file_name(file):
    return os.path.basename(file)

def get_extension(file):
    basename = os.path.basename(file)
    pos = basename.rfind('.')
    if pos == -1:
        return ''
    extension = basename[pos + 1:]
    extension = extension.rstrip('\\')
    return extension.lower()

def remove_extension(file):
    pos = file.rfind('.')
    if pos == -1:
        return file
    else:
        return file[:pos]

def location_to_file_name(loc, temp_folder=""):
    result = loc
    for c in [':', '/', '\\', '.']:
        result = result.replace(c, '_')
    return temp_folder + result

def get_format(value):
    if isinstance(value, (int, np.int32, np.uint8)):
        return 'i'
    elif isinstance(value, (float, np.float32, np.float64)):
        return 'f'
    else:
        raise TypeError(f"Unsupported type: {type(value)}")
        
def extract_unique_parts(strings):
    if not strings:
        return strings
    # Step 1: Split each string by '_' and '-'
    split_strings = [re.split(r'[-_]', s) for s in strings]
    # Step 2: Transpose the list to identify common elements **at the same position**
    transposed = list(zip(*split_strings))
    # Step 3: Identify indices where all elements are identical
    common_indices = {i for i, values in enumerate(transposed) if len(set(values)) == 1}
    # Step 4: Remove elements at common indices
    unique_parts = [
        [value for i, value in enumerate(parts) if i not in common_indices]
        for parts in split_strings
    ]
    # Step 5: Reconstruct the unique strings
    return ['_'.join(parts) for parts in unique_parts]