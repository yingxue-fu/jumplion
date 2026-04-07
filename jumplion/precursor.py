# builtin
from dataclasses import dataclass, field
# external
import numpy as np
from numba import types, typed, njit
# local
from jumplion.utils import proton, OH, fTypeB, fTypeY, Loss, loss_none, loss_N

#
GeneratorAccuracy = 5.0  # generator mass accuracy
ForceFragRec = False
MinFrAAs = 3
MaxRecCharge = 4
MaxRecLoss = 4

class Precursor:
    def __init__(self):
        self.index = 0
        
        self.mz = 0.0
        self.charge = 0
        
        self.iRT = 0.0
        self.iIM = 0.0
        
        self.nAAs = 0
        self.products = []

    def init(self, index, mz, charge, iRT):
        self.index = index
        self.mz = mz
        self.charge = charge
        self.iRT = iRT
        
    def free(self):
        self.products.clear()  

class Product:
    def __init__(self, mz=0.0, height=0.0, charge=0, ion_type=0, index=0, loss=0):
        self.mz = mz
        self.height = height      # intensity
        self.charge = charge
        self.ion_type = ion_type  # b, y ions
        self.index = index        # number of AAs to the left from the cleavage site (i.e., indexing based on b ions)
        self.loss = loss          # index of loss type

    def __lt__(self, other):
        return self.mz < other.mz

    def ion_code(self):
        loss_other = 5
        return ((((self.ion_type * 20 + self.charge)) * (loss_other + 1) + self.loss) * 100 + self.index + 1)

# the numbers represent ion_type, index, loss, charge, mz
ion_tuple = types.Tuple((types.int64, types.int64, types.int64, types.int64, types.float64))

@dataclass(order=True)
class MOD:
    name: str = field(compare=True)
    aas: str = field(default="", compare=False)
    mass: float = field(default=0.0, compare=True)
    label: int = field(default=0, compare=False)
    
    def init(self, name: str, aas: str, mass: float, label: int = 0):
        self.name = name
        self.aas = aas
        self.mass = mass
        self.label = label

# List of Modifications
Modifications = [
    MOD("UniMod:4", 57.021464),
    MOD("Carbamidomethyl (C)", 57.021464),
    MOD("Carbamidomethyl", 57.021464),
    MOD("CAM", 57.021464),
    MOD("+57", 57.021464),
    MOD("+57.0", 57.021464),
	MOD("UniMod:35", 15.994915),
	MOD("Oxidation (M)", 15.994915),
	MOD("Oxidation", 15.994915),
	MOD("Oxi", 15.994915),
	MOD("+16", 15.994915),
	MOD("+16.0", 15.994915),
    MOD("UniMod:1", 42.010565),
	MOD("Acetyl (Protein N-term)", 42.010565),
	MOD("+42", 42.010565),
	MOD("+42.0", 42.010565),
]

OriginalModNames = False
def init_unimod():
    indices = set()
    mods = Modifications
    mods.sort()
    UniMod = [[None, None] for _ in range(len(mods))]
    UniModIndex = [[None, None] for _ in range(len(mods))]

    for i, mod in enumerate(mods):
        UniMod[i][0] = UniModIndex[i][0] = mod.name
        if mod.name.startswith("UniMod"):
            UniMod[i][1] = mod.name
            UniModIndex[i][1] = i
            indices.add(i)
        else:
            min_delta = float('inf')
            min_index = 0
            if not OriginalModNames:
                for j, other_mod in enumerate(mods):
                    if j != i:
                        if not other_mod.name.startswith("UniMod"):
                            continue
                        delta = abs(mod.mass - other_mod.mass)
                        if delta < min_delta:
                            min_delta = delta
                            min_index = j

            if min_delta > 0.00001:
                if not OriginalModNames:
                    print(f"Cannot find a UniMod modification match for {mod.name}: {min_delta} minimal mass discrepancy; using the original modification name")
                UniMod[i][1] = UniMod[i][0]
                UniModIndex[i][1] = i
                indices.add(i)
            else:
                UniMod[i][1] = mods[min_index].name
                UniModIndex[i][1] = min_index
                indices.add(min_index)

    UniModIndices = list(indices)
    
    return UniMod, UniModIndex, UniModIndices

UniMod, UniModIndex, UniModIndices = init_unimod()

# Amino acid indices based on their order in a predefined string
AAs = "GAVLIFMPWSCTYHKRQEND"

def init_AA_array():
    # Molecular weights of amino acids
    AA = {
        'G': 57.021464, 'A': 71.037114, 'V': 99.068414, 'L': 113.084064, 'I': 113.084064,
        'F': 147.068414, 'M': 131.040485, 'P': 97.052764, 'W': 186.079313, 'S': 87.032028,
        'C': 103.009185, 'T': 101.047679, 'Y': 163.06332, 'H': 137.058912, 'K': 128.094963,
        'R': 156.101111, 'Q': 128.058578, 'E': 129.042593, 'N': 114.042927, 'D': 115.026943,
        'U': 150.95363, 'X': 196.995499
    }

    AA_index = {aa: idx for idx, aa in enumerate(AAs)}
    AA_index['U'] = AA_index['C']  # Assigning index of Cysteine to Selenocysteine
    AA_index['X'] = AA_index['M']  # Assigning index of Methionine to 'X'
    
    return AA, AA_index

AA, AA_index = init_AA_array()

def pep_name(pr_name):
    """
    Returns the prefix of the input string before any digits.
    Removes any trailing digits from the input string.
    """
    result = pr_name
    for i in range(len(result) - 1, -1, -1):
        if not result[i].isdigit():
            break
    result = result[:i + 1]
    return result

def closing_bracket(name, symbol, pos):
    """
    Finds the index of the closing bracket corresponding to the first opening bracket
    found at position 'pos' in the string 'name'. This function handles nested brackets.
    
    Args:
    name (str): The string containing brackets.
    symbol (char): The opening bracket character, either '(' or '['.
    pos (int): The position in 'name' where the opening bracket is located.
    
    Returns:
    int: The index of the matching closing bracket.
    """
    if symbol == '(':
        close_symbol = ')'
    elif symbol == '[':
        close_symbol = ']'
    else:
        raise ValueError("Unsupported bracket type")

    par = 1  # Initialize parenthesis/bracket nesting level
    for end in range(pos + 1, len(name)):
        if name[end] == close_symbol:
            par -= 1
            if par == 0:
                return end  # Return the position of the matching closing bracket
        elif name[end] == symbol:
            par += 1

    return len(name)  # Return the length of the string if no closing bracket is found


def get_aas(name):
    """
    Extracts amino acids from the given string, ignoring modifications or parts
    of the string enclosed in brackets or parentheses.
    
    Args:
    name (str): The string containing amino acids possibly intermixed with modifications.
    
    Returns:
    str: A string of amino acids extracted from 'name'.
    """
    result = []
    i = 0
    while i < len(name):
        symbol = name[i]
        if 'A' <= symbol <= 'Z':
            result.append(symbol)
            i += 1
        elif symbol in '([':
            i = closing_bracket(name, symbol, i)
            if i == -1:
                break  # no closing bracket found, stop processing
            i += 1  # move past the closing bracket
        else:
            i += 1  # ignore the current character and move on

    return ''.join(result)

class SequenceError(Exception):
    pass


def get_sequence_masses(name):
    
    """ calculate the mass of amino acid residues given a peptide sequence. """
    
    result = []
    add = 0.0
    
    i = 0
    while i < len(name):
        symbol = name[i]
        # When symbol is not an amino acid (i.e., modifications)
        if symbol < 'A' or symbol > 'Z':
            if symbol in '([':
                
                start = i + 1
                end = closing_bracket(name, symbol, i) # position of closing bracket
                mod = name[start:end]
                
                found_mod = False
                for mod_obj in Modifications:
                    if mod_obj.name == mod:
                        found_mod = True
                        # add modification mass
                        if not result: # for the first amino acid
                            add += mod_obj.mass
                        else:
                            result[-1] += mod_obj.mass
                            
                        break
                    
                if not found_mod:
                    raise SequenceError(f"Unknown modification: {mod}")
                    
                i = end
            i += 1
            continue
        # When symbol is an amino acid
        result.append(AA.get(symbol, 0.0) + add)
        add = 0.0
        i += 1
    
    return np.array(result, dtype = np.float32)


def peptide_length(name):
    """
    Calculates the length of the peptide sequence, ignoring modification annotations.
    """
    i = 0
    n = 0
    while i < len(name):
        symbol = name[i]
        if 'A' <= symbol <= 'Z':  # Check if the character is an amino acid
            n += 1
            i += 1
        elif symbol in '([':
            i = closing_bracket(name, symbol, i) + 1
        else:
            i += 1
    return n

def missed_KR_cleavages(name: str) -> int:
    n = 0
    last = 0
    i = 0
    while i < len(name) - 1:
        symbol = name[i]
        if symbol not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if symbol in '([':
                i = closing_bracket(name, symbol, i)
            i += 1
            continue
        if symbol in 'KR':
            n += 1
            last = 1
        else:
            last = 0
        i += 1
    return n - last

def get_mod_state(name, length):
    """
    Generates a list indicating whether each amino acid position in a peptide is modified.
    """
    result = [False] * length
    i = 0
    j = 0  # index for amino acids
    while i < len(name):
        symbol = name[i]
        if 'A' <= symbol <= 'Z':
            if j < length:  # Increment j only if within the peptide length
                j += 1
            i += 1
        elif symbol in '([':
            if j < length:  # Mark this position as modified if within bounds
                result[j] = True
            i = closing_bracket(name, symbol, i) + 1
        else:
            i += 1
    return result

def to_canonical(name):
    result = ""
    i = 0
    while i < len(name):
        symbol = name[i]
        if symbol < 'A' or symbol > 'Z':
            if symbol != '(' and symbol != '[':
                i += 1
                continue
            i += 1
            end = closing_bracket(name, symbol, i)
            if end == -1:
                raise Exception("incorrect peptide name format: " + name)

            mod = name[i:end]
            for j in range(len(UniMod)):
                if UniMod[j][0] == mod:
                    result += "(" + UniMod[j][1] + ")"
                    break
            if j == len(Modifications):
                result += "(" + mod + ")"  # no warning here
            i = end
            continue
        else:
            result += symbol
        i += 1
    return result

def to_canonical_with_charge(name, charge):
    return to_canonical(name) + str(charge)

def to_eg(name):
    eg = ""
    i = 0
    while i < len(name):
        symbol = name[i]
        if symbol < 'A' or symbol > 'Z':
            if symbol != '(' and symbol != '[':
                i += 1
                continue
            i += 1
            end = closing_bracket(name, symbol, i)
            if end == -1:
                raise Exception("incorrect peptide name format: " + name)

            mod = name[i:end]
            for j in range(len(Modifications)):
                if Modifications[j].name == mod:
                    if not Modifications[j].label:
                        eg += "(" + UniMod[j][1] + ")"
                    break
            if j == len(Modifications):
                eg += "(" + mod + ")"  # no warning here
            i = end
            continue
        else:
            eg += symbol
        i += 1
    return eg

def to_charged_eg(name, charge):
    return to_eg(name) + str(charge)


@njit
def generate_products(sequence, charge, loss, cnt_ref, min_aas=0):
    '''
    Generate products with a specific neutral loss type.

    Parameters
    ----------
    sequence : TYPE
        DESCRIPTION.
    charge : TYPE
        DESCRIPTION.
    loss : TYPE
        DESCRIPTION.
    cnt_ref : TYPE
        DESCRIPTION.
    min_aas : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    result : List
        A list of products stored in tuples.

    '''
    
    c = types.float32(charge) # Convert charge to float 
    
    s = np.sum(sequence)         # Sum of the full sequence masses
    curr = types.float32(0.0) # Track cumulative mass for b ions
    
    result = typed.List.empty_list(ion_tuple)
    
    cnt = types.int32(0)  # to keep track of number of products generated
    for i in range(len(sequence) - 1):
        curr += sequence[i]
        # mass
        b = curr                    # mass sum for type B ion, N-terminal, left side
        y = s - curr + proton + OH  # mass sum for type Y ion, C-terminal, right side
        # m/z
        if i + 1 >= min_aas:
            b_mz = (b + c * proton - Loss[loss]) / c  # m/z for b ion
            ion_b = (fTypeB, i + 1, loss, charge, np.round(b_mz, 5))
            result.append(ion_b) # Append b ion
            cnt += 1
        if len(sequence) - i - 1 >= min_aas:
            y_mz = (y + c * proton - Loss[loss]) / c  # m/z for y ion
            ion_y = (fTypeY, i + 1, loss, charge, np.round(y_mz, 5))
            result.append(ion_y) # Append y ion
            cnt += 1

    cnt_ref[0] = cnt  # update the external count reference
    return result

@njit
def recognise_products_by_mass(
        sequence, 
        products, 
        pr_charge, 
        full_spectrum = False, 
        max_charge = 19, 
        loss_cap = loss_N):
    '''
    Parameters
    ----------
    sequence : TYPE
        DESCRIPTION.
    products : List
        A list of products stored in tuples.
    pr_charge : TYPE
        DESCRIPTION.
    full_spectrum : TYPE, optional
        DESCRIPTION. The default is False.
    max_charge : TYPE, optional
        DESCRIPTION. The default is 19.
    loss_cap : TYPE, optional
        DESCRIPTION. The default is loss_N.

    Returns
    -------
    result : List 
        A list of products stored in tuples.
    gen_acc : float
        Final mass accuracy used in mass comparsion.
    bool
        A flag indicating not all products were recognized when it's True.

    '''
    max_gen_acc = 0.0
    #result = [(0, 0, 0, 0, 0.0) for _ in range(len(products))]
    result = typed.List.empty_list(ion_tuple)
    for _ in range(len(products)):
        result.append((0, 0, 0, 0, 0.0))
    v = typed.List.empty_list(ion_tuple)
    s = np.sum(sequence)
    
    gen_acc = GeneratorAccuracy  # defult: 5 ppm

    while True:
        cnt = np.array([0], dtype=np.int32)
        tot = 0
        index = 0
        charge = 1
        loss = loss_none
        
        for i in range(len(result)):
            result[i] = (0, 0, 0, 0, 0.0)

        while charge <= max_charge:
            # generate theoretical products for current sequence
            v = generate_products(sequence, charge, loss, cnt)
            if charge == 1:
                mz = (s + OH + (1.0 + pr_charge) * proton - Loss[loss]) / pr_charge
                v.append((fTypeY, len(sequence), loss, pr_charge, np.round(mz, 5)))  # non-producted
            # check the given products
            for i in range(len(products)):
                if result[i][3] != 0:  # Check charge (index 3):
                    continue
                # check m/z
                frag_mz = products[i][4]  # mz is at index 4
                margin = frag_mz * gen_acc
                min_delta = margin
                # compare with theoretical m/z
                for j in range(len(v)):
                    ion = v[j]
                    delta = abs(ion[4] - frag_mz)
                    if delta < min_delta:
                        min_delta = delta
                        index = j
                # collect products with m/z under the mass tolerance
                if min_delta < margin:
                    tot += 1
                    result[i] = v[index]
                    if tot >= len(products):
                        break

            if tot >= len(products):
                break

            loss += 1
            if loss == loss_cap:
                loss = loss_none
                charge += 1

        if tot >= len(products):
            break

        if full_spectrum:
            return result, gen_acc, False # Return current gen_acc and a flag indicating not all products were recognized

        gen_acc *= 2.0
        if gen_acc > max_gen_acc:
            max_gen_acc = gen_acc
        v = typed.List.empty_list(ion_tuple) # Clear the list
        
    return result, gen_acc, gen_acc > GeneratorAccuracy


def annotate_products_charges(sequence, products, pr_charge):
    
    if not len(sequence):
        return
    
    products_tuples = typed.List.empty_list(ion_tuple)
    for f in products:
        products_tuples.append((f.ion_type, f.index, f.loss, f.charge, f.mz))
    pattern, _, _ = recognise_products_by_mass(sequence, products_tuples, pr_charge, False, MaxRecCharge, MaxRecLoss)

    for i, product in enumerate(pattern):
        products[i].charge = product[3]
    
    return products