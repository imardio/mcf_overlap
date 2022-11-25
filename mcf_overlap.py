"""
MCFO: Mode Composition Factor Overlaps

A program to gauge the overlap between vibrational modes based on their kinetic energy distributions (KED)
Written by: Mauricio Maldonado-Dominguez

The theory behind is described in: 
Phys. Chem. Chem. Phys., 2019, 21, 24912-24918.

Applications of mode composition factors to the analysis of nonequilibrium reactivity:
J. Am. Chem. Soc. 2020, 142, 8, 3947-3958.
Chem. Sci. 2021,

Improvements to be done:

1) Mode overlap with plotting for a single structure. DONE
2) Overlaps between RC-TS and TS-PC (DONE)
3) Check if atomic numbers are the same in all cases
4) Maximize overlaps
   a) keep x-axis as is
   b) get indices for the maximum value in the last row of every overlap matrix.
   c) build a vector with the indices from b.
   d) build the overlap matrix not with range(), but with the vector from c.
   e) plot and be happy

NOTES (28.07.2021)

It seems to work up to (4e), without complaining. However, the following points need to be checked.

    a) Print the vector from point 4c for inspection. Not sure if it is currently correct.
    b) There are, likely, several modes sharing a 'maximum overlap mode'. We must rank them.
    c) To achieve (b), we need to store also the vector with the overlap values, not only the indices. Or a dataframe containing both.
    d) Alternatively, d(nu) can be used as a criterion for prioritization.

"""


########
#Modules
########

import pandas as pd
import argparse
import os
import sys
import logging
from datetime import datetime
import itertools
import time
from io import StringIO
import re
from shutil import copy as cp
import glob
import matplotlib.pyplot as plt

########
#Dictionaries 
########

atomic_symbol = {
    1:'H', 2:'He', 3:'Li', 4:'Be', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 10:'Ne',
    11:'Na', 12:'Mg', 13:'Al', 14:'Si',15:'P',16:'S',17:'Cl',18:'Ar',19:'K',
    20:'Ca', 21 : "Sc" , 22 : "Ti" , 23 : "V"  , 24 : "Cr" , 25 : "Mn", 26 : "Fe" ,
    27: "Co" , 28 : "Ni" , 29 : "Cu" , 30 : "Zn", 31 : "Ga" , 32 : "Ge" , 33 : "As" ,
    34: "Se" , 35 : "Br", 36 : "Kr" , 37 : "Rb" , 38 : "Sr" , 39 : "Y"  , 40 : "Zr",
    41: "Nb" , 42 : "Mo" , 43 : "Tc" , 44 : "Ru" , 45 : "Rh", 46 : "Pd" , 47 : "Ag" ,
    48: "Cd" , 49 : "In" , 50 : "Sn", 51 : "Sb" , 52 : "Te" , 53 : "I"  , 54 : "Xe" ,
    55: "Cs", 56 : "Ba" , 57 : "La" , 58 : "Ce" , 59 : "Pr" , 60 : "Nd", 61 : "Pm" ,
    62: "Sm", 63 : "Eu" , 64 : "Gd" , 65 : "Tb", 66 : "Dy" , 67 : "Ho" , 68 : "Er" ,
    69: "Tm", 70 : "Yb", 71 : "Lu" , 72 : "Hf" , 73 : "Ta" , 74 : "W"  , 75 : "Re",
    76: "Os", 77 : "Ir" , 78 : "Pt" , 79 : "Au" , 80 : "Hg", 81 : "Tl" , 82 : "Pb" ,
    83: "Bi",
    }

atomic_mass = {
    1:1.0, 2:4.0, 3:7.0, 4:9.0, 5:11.0, 6:12.0, 7:14.0, 8:16.0, 9:19.0, 10:20.0,
    11:23.0, 12:24.0, 13:27.0, 14:28.0, 15:31.0, 16:32.0, 17:35.0, 18:40.0, 19:39.0, 
    20:40.0, 21:45.0, 22:48.0, 23:51.0, 24:52.0, 25:55.0, 26:56.0, 27:59.0, 28:59.0, 
    29:64.0, 30:65.0, 31:70.0, 32:73.0, 33:75.0, 34:79.0, 35:80.0, 36:84.0, 37:85.0, 
    38:88.0, 39:89.0, 40:91.0, 41:93.0, 42:96.0, 43:97.0, 44:101.0, 45:104.0, 46:106.0, 
    47:108.0, 48:112.0, 49:115.0, 50:119.0, 51:122.0, 52:128.0, 53:127.0, 54:131.0,
    55:133.0, 56:137.0,
    }

#######
# Program
#######

def rmcf(filename1, filename2, filename3):

    logging.basicConfig(filename='KED_overlaps_' + filename2[:-4] + '.LOG', level=logging.DEBUG, format='%(message)s', filemode='w')

    mpl_logger = logging.getLogger('plt')
    mpl_logger.setLevel(logging.WARNING)

    datetime_now = datetime.now()
    formatted_datetime = datetime_now.strftime("%Y %b %d %H:%M:%S")

    logging.info('#####################################################################\n')
    logging.info('MCFO: Mode Composition Factor Overlaps\n')
    logging.info('A program to gauge the overlap between vibrational modes based on their kinetic energy distributions (KED)')
    logging.info('Written by: Mauricio Maldonado-Dominguez')
    logging.info('J. Heyrovsky Institute of Physical Chemistry of the CAS')
    logging.info('Run date: '+formatted_datetime+'\n')
    logging.info('#####################################################################\n')

    files = [filename1, filename2, filename3]

    for file in files:

        home = os.getcwd()

        scratch(file)

        print('Analyzing ' + file)

        logging.info('Reading Gaussian output: ' + file)
        check_success(file)

        if "NAtoms=" in open(file,'rt').read():
            n_atoms_1(file)
        elif "Input orientation:" in open(file,'rt').read():
            n_atoms_2(file)
        elif "Standard orientation:" in open(file,'rt').read():
            n_atoms_3(file)
        else:
            logging.info('The program aborted with Error 1. Unrecognized format in Gaussian output file.' + '\n')
            print("Error 1. Unrecognized format in Gaussian output file. Aborting")
            exit()

        logging.info('The number of atoms for ' + file + 'is: ' + str(total_atoms) + '\n')

        if "Input orientation" in open(file,'rt').read():  

            last_input(file)

            logging.info('The last geometry for ' + file + ' is:' + '\n')
            logging.info(last_xyz.to_string(header=True, index=True) + '\n')

        elif "Standard orientation:" in open(file,'rt').read():

            last_standard(file)

            logging.info('The last geometry for ' + file + ' is:' + '\n')
            logging.info(last_xyz.to_string(header=True, index=True) + '\n')

        else:

            logging.info('The program aborted with Error 2. Unrecognized format in file ' + file + '\n')
            print('Error 2. Unrecognized format in file ' + file + '. Aborting')
            exit()

        if "atoms frozen in the vibrational analysis" in open(file, 'rt').read():
            logging.info('Frozen atoms found in file ' + file + '. This is currently unsupported. Aborting MCF analysis.' + '\n')
            print('Frozen atoms found in file ' + file + '. Aborting the RMCF analysis.')
            exit()
        elif "Harmonic frequencies" in open(file, 'rt').read():
            logging.info('Harmonic frequencies found in file ' + file + '. The program will now calculate mode composition factors.')

            normal_modes(file)
            split_modes()

            for value in range(1,1000):
                if os.path.isfile('./modes_' + str(value)):
                    delete_lines('modes_' + str(value), [0,1,2,3,4,5,6])
                else:
                    break

            split_finer()
            ked()

# Implement KED for all modes
            letter=["a","b","c"]
            for value in range(1,int(total_atoms)-1):
                for mode in range(len(letter)):
                    if os.path.isfile('./mode_' + str(value) + '_' + letter[mode]):
                        ked1('./mode_' + str(value) + '_' + letter[mode])
                    else:
                        break

            logging.info('Composition factors for all modes in file ' + file + ' were calculated.' + '\n')

            rename_mcf()

            normalization()

#            overlap(file)

        else:
            print('Error 3. Normal vibrational modes not found in file ' + file + '. Aborting' + '\n')
            logging.info('The program aborted with Error 3. Normal vibrational modes not found in file ' + file + '\n')
            exit()

        remove_aux()

        print('Success!')

        datetime_end = datetime.now()
        formatted_endtime = datetime_end.strftime("%Y %b %d %H:%M:%S")

        logging.info('MCFO analysis of file ' + file + ' ended on: ' + formatted_endtime + '\n')

        os.chdir(home)

    home = os.getcwd()

    scratch_cross(filename2)

    cross_overlap(filename1, filename2, filename3, home)


########
#Functions
########

def check_success(filename):
    with open(filename) as f:
        if (' Normal termination of Gaussian') in f.read():
            logging.info('The Gaussian calculation for ' + filename + ' terminated normally.'+'\n')
        else: 
            logging.info('The Gaussian calculation for ' + filename + ' did not finish correctly. Aborting.'+'\n')
            print('Error. Please check the Gaussian output for' + filename)
            exit()

def scratch(filename):
    temp='./temp_'+filename
    if not os.path.exists(temp):
        os.mkdir(temp)
        cp(filename,temp + '/' + filename)
        os.chdir(temp)
    else:
        print('Please remove the ' + temp + ' directory manually. Aborting.')
        exit()

def scratch_cross(filename):
    temp='./cross_overlaps_' + filename[:-4]
    if not os.path.exists(temp):
        os.mkdir(temp)
        os.chdir(temp)
    else:
        print('Please remove the ' + temp + ' directory manually. Aborting.')
        exit()

####Extract the total number of atoms as an integer, without resorting to the "NAtoms" identifier.

def n_atoms_1(filename):
    global total_atoms
    if "NAtoms" in open(filename,'rt').read():
        with open(filename,'rt') as f:
            for line in f.readlines():
                if "NAtoms" in line:
                    if re.split(r'\s',line)[6].rstrip('\n') is not "":
#                        If true, we found the number of atoms at position 6
                        total_atoms = re.split(r'\s',line)[6].rstrip('\n')
                        with open('NAtoms', 'w') as atoms:
                            atoms.write(str(total_atoms))
                        break
                    elif re.split(r'\s',line)[8].rstrip('\n') is not "":
#                        If true, we found the number of atoms at position 8
                        total_atoms = re.split(r'\s',line)[8].rstrip('\n')
                        with open('NAtoms', 'w') as atoms:
                            atoms.write(str(total_atoms))
                        break
              
def n_atoms_2(filename):
    global total_atoms
    with open(filename,'rt') as file: 
        for line in file:
            lines.append(str(line.rstrip('\n'))) 
    index1 = lines.index(str("Standard orientation:"))
    aux_index = []
    for line in lines:
        substring = line[:21]
        if substring == " Rotational constants":
            index3 = lines.index(line)
            aux_index.append(index3)
        else:
            pass
    index2 = aux_index[0]
    input_orientation = lines[index1 + 5:index2 - 1]
    total_atoms = len(input_orientation)
    with open('NAtoms', 'w') as atoms:
        atoms.write(str(total_atoms))

def n_atoms_3(filename):
    global total_atoms
    with open(filename,'rt') as file: 
        for line in file:
            lines.append(str(line.rstrip('\n'))) 
    index1 = lines.index(str("Input orientation:"))
    aux_index = []
    for line in lines:
        substring = line[:21]
        if substring == " Rotational constants":
            index3 = lines.index(line)
            aux_index.append(index3)
        else:
            pass
    index2 = aux_index[0]
    input_orientation = lines[index1 + 5:index2 - 1]
    total_atoms = len(input_orientation)
    with open('NAtoms', 'w') as atoms:
        atoms.write(str(total_atoms))

#### Extract normal modes

def normal_modes(filename):
    global lines
    lines = []
    with open(filename,'rt') as file: 
        for line in file:
            lines.append(str(line.rstrip('\n'))) 
    index1 = lines.index(str(" and normal coordinates:"))
    index2 = lines.index(str(" - Thermochemistry -"))
    list = lines[index1 + 1:index2 - 2]
    with open('modes', 'w') as file:
        for line in list:
            file.writelines(str(line) + '\n')

#### Cut the 'modes' file into 3-mode chunks 

def split_modes():
    total_lines = int(total_atoms) + 7
    smallfile = None
    with open('modes') as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % total_lines == 0:
                if smallfile:
                    smallfile.close()
                small_filename = 'modes_{}'.format(int((lineno + total_lines)/(total_lines)))
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

# split each chunk into individual modes

def split_finer():
    for value in range(1,int(total_atoms)-1):
        mode1_xyz = pd.read_table('modes_' + str(value), usecols=[2,3,4], delim_whitespace=True, header=None)
        mode_atoms = pd.read_table('modes_' + str(value), usecols=[1], delim_whitespace=True, header=None)
        mode1_xyz.insert(0, 1, mode_atoms)
        mode1_xyz.index += 1

        filename = 'mode_' + str(value) + '_a'
        with open (filename, "w") as file:
            file.write(mode1_xyz.to_string(header=None, index=True))

        mode2_xyz = pd.read_table('modes_' + str(value), usecols=[5,6,7], delim_whitespace=True, header=None)
        mode_atoms = pd.read_table('modes_' + str(value), usecols=[1], delim_whitespace=True, header=None)
        mode2_xyz.insert(0, 1, mode_atoms)
        mode2_xyz.index += 1

        filename = 'mode_' + str(value) + '_b'
        with open (filename, "w") as file:
            file.write(mode2_xyz.to_string(header=None, index=True))

        mode3_xyz = pd.read_table('modes_' + str(value), usecols=[8,9,10], delim_whitespace=True, header=None)
        mode_atoms = pd.read_table('modes_' + str(value), usecols=[1], delim_whitespace=True, header=None)
        mode3_xyz.insert(0, 1, mode_atoms)
        mode3_xyz.index += 1

        filename = 'mode_' + str(value) + '_c'
        with open (filename, "w") as file:
            file.write(mode3_xyz.to_string(header=None, index=True))

# Remove all headers

def delete_lines(original_file, line_numbers):
    is_skipped = False
    counter = 0
    # Create name of dummy / temporary file
    dummy_file = original_file + '.bak'
    # Open original file in read only mode and dummy file in write mode
    with open(original_file, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        for line in read_obj:
            # If current line number exist in list then skip copying that line
            if counter not in line_numbers:
                write_obj.write(line)
            else:
                is_skipped = True
            counter += 1

    if is_skipped:
        os.remove(original_file)
        os.rename(dummy_file, original_file)
    else:
        os.remove(dummy_file)

#### Calculate KED. Currently implemented only for the reactive mode. 
#### Future implementations will address real modes.

def ked():
    global mode_xyz
    mode_xyz = pd.read_table('modes_1', usecols=[2,3,4], delim_whitespace=True, header=None)
    mode_atoms = pd.read_table('modes_1', usecols=[1], delim_whitespace=True, header=None)
    mode_symbols = pd.read_table('modes_1', usecols=[1], delim_whitespace=True, header=None)
    mode_symbols.replace(atomic_symbol, inplace=True)
    mode_atoms.replace(atomic_mass, inplace=True)
    mode_masses = pd.to_numeric(mode_atoms[1], errors='coerce')
    norm2 = (mode_xyz**2).sum(axis=1)
    for k, v in atomic_mass.items():
        atomic_mass[k] = float(v)
    preKED = norm2*mode_masses
    normalize = preKED.sum()
    KED = preKED.div(normalize)
    mode_xyz[5] = KED
    mode_xyz.insert(0, 1, mode_symbols)
    mode_xyz.columns=['Atom', 'dx', 'dy', 'dz', 'KED']
    mode_xyz.index += 1

def ked1(chunk):
    mode1_xyz = pd.read_table(chunk, usecols=[2,3,4], delim_whitespace=True, header=None)
    mode_atoms = pd.read_table(chunk, usecols=[1], delim_whitespace=True, header=None)
    mode_symbols = pd.read_table(chunk, usecols=[1], delim_whitespace=True, header=None)
    mode_symbols.replace(atomic_symbol, inplace=True)
    mode_atoms.replace(atomic_mass, inplace=True)
    mode_masses = pd.to_numeric(mode_atoms[1], errors='coerce')
    norm2 = (mode1_xyz**2).sum(axis=1)
    for k, v in atomic_mass.items():
        atomic_mass[k] = float(v)
    preKED = norm2*mode_masses
    normalize = preKED.sum()
    KED = preKED.div(normalize)
    mode1_xyz[5] = KED
    mode1_xyz.insert(0, 1, mode_symbols)
    mode1_xyz.columns=['Atom', 'dx', 'dy', 'dz', 'KED']
    mode1_xyz.index += 1
    filename = chunk + '_mcf'
    with open (filename, "w") as file:
        file.write(mode1_xyz.to_string(header=True, index=True))

def normalization():
    ked_all = []
    for number in range(1,3*int(total_atoms)-5):
        data = pd.read_table('mode_' + str(number), usecols=[5], delim_whitespace=True, header=None, skiprows=[0])
        ked_all.append(data)
    ked_all = pd.concat(ked_all,axis=1)
    ked_all['Total'] = ked_all.sum(axis=1)
    with open ('ked_all_modes', "w") as file:
        file.write(ked_all.to_string(header=False, index=False))

def rename_mcf():
    for count, file in enumerate(glob.glob('mode_*_mcf')):
        dst = 'mode_' + str(int(count) + 1)
        os.rename(file, dst)

def overlap(name):
    print('Calculating overlaps')
    table = pd.read_table('ked_all_modes', delim_whitespace=True, header=None, usecols = [i for i in range(0,3*int(total_atoms)-6)])
    for mode_A in range(0,3*int(total_atoms)-6):
        overlaps = []
        for mode_B in range(0,3*int(total_atoms)-6):
            diff = table[mode_A] - table[mode_B]        
            overlaps.append(diff)
        overlaps = pd.concat(overlaps,axis=1)
        overlaps.columns = range(len(overlaps.columns))
        overlaps.loc['Total'] = 1- 0.5*overlaps.abs().sum()

        with open ('overlaps_'+ str(mode_A + 1), "w") as file:
            file.write(overlaps.abs().to_string(header=False, index=False))

    overlap_matrix = []
    for mode_A in range(0,3*int(total_atoms)-6):
        over_mode = pd.read_table('overlaps_'+ str(mode_A + 1), delim_whitespace=True, header=None).iloc[-1:]
        overlap_matrix.append(over_mode)
    overlap_matrix = pd.concat(overlap_matrix,axis=0)

    with open ('overlap_matrix', "w") as file:
        file.write(overlap_matrix.to_string(header=False, index=False))

    plt.imshow(overlap_matrix, cmap='Greys', interpolation='nearest')
    plt.savefig(str(name) + '.png')

def cross_overlap(file1, file2, file3, house):
    print('Calculating overlaps between ' + file1 + ' and ' + file2)
    temp1 = house + '/temp_' + file1
    temp2 = house + '/temp_' + file2
    temp3 = house + '/temp_' + file3

# Column range is specified because the last column (the row sums) must be omitted from overlap calculations

    table1 = pd.read_table(temp1 + '/ked_all_modes', delim_whitespace=True, header=None, usecols = [i for i in range(0,3*int(total_atoms)-6)])
    table2 = pd.read_table(temp2 + '/ked_all_modes', delim_whitespace=True, header=None, usecols = [i for i in range(0,3*int(total_atoms)-6)])
    table3 = pd.read_table(temp3 + '/ked_all_modes', delim_whitespace=True, header=None, usecols = [i for i in range(0,3*int(total_atoms)-6)])

### FIRST, OVERLAPS BETWEEN RC AND TS

    max_over = []

    for mode_A in range(0,3*int(total_atoms)-6):
        overlaps = []
        totals = []
        for mode_B in range(0,3*int(total_atoms)-6):
            diff = table1[mode_A] - table2[mode_B]        
            overlaps.append(diff)
        overlaps = pd.concat(overlaps,axis=1)
        overlaps.columns = range(len(overlaps.columns))
        overlap_sums = 1 - 0.5*overlaps.abs().sum()

#new lines from here

        totals.append(overlap_sums)
        overlaps = pd.concat(totals,axis=0)
        print(overlap_sums)

## new lines start here

        max = overlap_sums.idxmax(axis=0)
#        max2 = overlaps.loc['Total'].max()

        max_over.append(max)
#        max_over.append(max2)

#    max_over = pd.concat(max_over,axis=0)

#    with open ('max_overlaps', "w") as file:
#        file.write(max_over.to_string(header=False, index=False))


##new lines end here

        with open ('overlaps_'+ str(mode_A + 1), "w") as file:
            file.write(overlaps.abs().to_string(header=False, index=False))

    print(max_over)

    overlap_rc_ts = []
    for mode_A in range(0,3*int(total_atoms)-6):
        over_mode = pd.read_table('overlaps_'+ str(mode_A + 1), delim_whitespace=True, header=None).iloc[-1:]
        overlap_rc_ts.append(over_mode)
    overlap_rc_ts = pd.concat(overlap_rc_ts,axis=0)

    with open ('overlap_rc_ts', "w") as file:
        file.write(overlap_rc_ts.to_string(header=False, index=False))

    plt.imshow(overlap_rc_ts, cmap='Greys', interpolation='nearest')
    plt.savefig('overlap_rc_ts.png')

### NOW, LET'S MAXIMIZE OVERLAPS

    overlap_rc_ts_max = []

    for mode_A in max_over:
        over_mode = pd.read_table('overlaps_'+ str(mode_A + 1), delim_whitespace=True, header=None).iloc[-1:]
        overlap_rc_ts_max.append(over_mode)
    overlap_rc_ts_max = pd.concat(overlap_rc_ts_max,axis=0)

    with open ('overlap_rc_ts_max', "w") as file:
        file.write(overlap_rc_ts_max.to_string(header=False, index=False))

    plt.imshow(overlap_rc_ts_max, cmap='Greys', interpolation='nearest')
    plt.savefig('overlap_rc_ts_max.png')


### LAST, OVERLAPS BETWEEN PC AND TS

    print('Calculating overlaps between ' + file2 + ' and ' + file3)

    for mode_A in range(0,3*int(total_atoms)-6):
        overlaps = []
        for mode_B in range(0,3*int(total_atoms)-6):
            diff = table3[mode_A] - table2[mode_B]        
            overlaps.append(diff)
        overlaps = pd.concat(overlaps,axis=1)
        overlaps.columns = range(len(overlaps.columns))
        overlaps.loc['Total'] = 1- 0.5*overlaps.abs().sum()

        with open ('overlaps_'+ str(mode_A + 1), "w") as file:
            file.write(overlaps.abs().to_string(header=False, index=False))

    overlap_pc_ts = []
    for mode_A in range(0,3*int(total_atoms)-6):
        over_mode = pd.read_table('overlaps_'+ str(mode_A + 1), delim_whitespace=True, header=None).iloc[-1:]
        overlap_pc_ts.append(over_mode)
    overlap_pc_ts = pd.concat(overlap_pc_ts,axis=0)

    with open ('overlap_pc_ts', "w") as file:
        file.write(overlap_pc_ts.to_string(header=False, index=False))

    plt.imshow(overlap_pc_ts, cmap='Greys', interpolation='nearest')
    plt.savefig('overlap_pc_ts.png')

### EXTRACTING THE LAST GEOMETRY FROM THE OUTPUT FILE ###

def unique_geometry():
    with open('last_geometry', 'w') as unique:
        for element in input_orientation:
            unique.write(element + '\n')
    global only_xyz
    only_xyz = pd.read_table('last_geometry', usecols=[3,4,5], delim_whitespace=True, header=None)
    only_symbols = pd.read_table('last_geometry', usecols=[1], delim_whitespace=True, header=None)
    only_symbols.replace(atomic_symbol, inplace=True)
    only_xyz.insert(0, 1, only_symbols)
    only_xyz.columns=['Atom', 'X', 'Y', 'Z']
    only_xyz.index += 1

def last_input(filename):
    lines = []
    with open(filename,'rt') as file: 
        for line in file:
            lines.append(str(line.rstrip('\n'))) 
    input = "Input orientation:"
    locations = [] # Here we will compile where the geometry definitions begin throughout the output file
    with open(filename,'r') as f:
        for num, line in enumerate(f, 1):
            if input in line:
                a = (int(num) + 5)
                locations.append(str(a) + '\n')
            else:
                pass
    limit = " ---------------------------------------------------------------------\n"

    delimiters = [] # Here we will compile where the geometry definitions end throughout the output file
    for location in locations:
        with open(filename,'r') as f:
                skipped = itertools.islice(f, int(location), None)
                for num, line in enumerate(skipped, int(location) + 1):
                    if limit == line:
                        b = (int(num))
                        delimiters.append(str(b) + '\n')
                        break
                    else:
                        pass
    with open('geometry_locations','w') as output:
        for line in locations:
            output.writelines(str(line))
    with open('delimiter_locations','w') as output:
        for line in delimiters:
            output.writelines(str(line))
    with open('geometry_locations') as file:
        for line in file:
            pass
        index1 = int(line)
    with open('delimiter_locations') as file:
        for line in file:
            pass
        index2 = int(line)
    global last_xyz
    last_xyz = lines[index1 - 1:index2 - 1]
    with open('last_geometry', 'w') as file:
        for line in last_xyz:
            file.writelines(str(line) + '\n')
    last_xyz = pd.read_table('last_geometry', usecols=[3,4,5], delim_whitespace=True, header=None)
    geom_symbols = pd.read_table('last_geometry', usecols=[1], delim_whitespace=True, header=None)
    geom_symbols.replace(atomic_symbol, inplace=True)
    last_xyz.insert(0, 1, geom_symbols)
    last_xyz.columns=['Atom', 'X', 'Y', 'Z']
    last_xyz.index += 1

def last_standard(filename):
    lines = []
    with open(filename,'rt') as file: 
        for line in file:
            lines.append(str(line.rstrip('\n'))) 
    input = "Standard orientation:"
    locations = [] # Here we will compile where the geometry definitions begin throughout the output file
    with open(filename,'r') as f:
        for num, line in enumerate(f, 1):
            if input in line:
                a = (int(num) + 5)
                locations.append(str(a) + '\n')
            else:
                pass
    limit = " ---------------------------------------------------------------------\n"

    delimiters = [] # Here we will compile where the geometry definitions end throughout the output file
    for location in locations:
        with open(filename,'r') as f:
                skipped = itertools.islice(f, int(location), None)
                for num, line in enumerate(skipped, int(location) + 1):
                    if limit == line:
                        b = (int(num))
                        delimiters.append(str(b) + '\n')
                        break
                    else:
                        pass
    with open('geometry_locations','w') as output:
        for line in locations:
            output.writelines(str(line))
    with open('delimiter_locations','w') as output:
        for line in delimiters:
            output.writelines(str(line))
    with open('geometry_locations') as file:
        for line in file:
            pass
        index1 = int(line)
    with open('delimiter_locations') as file:
        for line in file:
            pass
        index2 = int(line)
    global last_xyz
    last_xyz = lines[index1 - 1:index2 - 1]
    with open('last_geometry', 'w') as file:
        for line in last_xyz:
            file.writelines(str(line) + '\n')
    last_xyz = pd.read_table('last_geometry', usecols=[3,4,5], delim_whitespace=True, header=None)
    geom_symbols = pd.read_table('last_geometry', usecols=[1], delim_whitespace=True, header=None)
    geom_symbols.replace(atomic_symbol, inplace=True)
    last_xyz.insert(0, 1, geom_symbols)
    last_xyz.columns=['Atom', 'X', 'Y', 'Z']
    last_xyz.index += 1

### Remove auxiliary files

def remove_aux():
    os.remove("delimiter_locations")
    os.remove("geometry_locations")
    os.remove("modes")
    os.remove("last_geometry")
    os.remove("NAtoms")
    for value in range(1,int(total_atoms)-1):
        os.remove("modes_" + str(value))
    letter=["a","b","c"]
    for value in range(1,int(total_atoms)-1):
        for mode in range(len(letter)):
            os.remove("mode_" + str(value) + "_" + letter[mode])

#######################
# Parsing the input and executing the code
#######################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program calculates the KED overlap a Gaussian TS calculation')
    parser.add_argument('GaussFile1', help='Gaussian output file (*.out or *.log) for RC')
    parser.add_argument('GaussFile2', help='Gaussian output file (*.out or *.log) for TS')
    parser.add_argument('GaussFile3', help='Gaussian output file (*.out or *.log) for PC')

    args = parser.parse_args()

    if not os.path.isfile(args.GaussFile1):
        print(args.GaussFile1 + " is not a valid Gaussian output file.")
        quit()

    if not os.path.isfile(args.GaussFile2):
        print(args.GaussFile2 + " is not a valid Gaussian output file.")
        quit()

    if not os.path.isfile(args.GaussFile3):
        print(args.GaussFile3 + " is not a valid Gaussian output file.")
        quit()

    rmcf(args.GaussFile1, args.GaussFile2, args.GaussFile3)
