# ligand_replacer
Pairfits a substructure into a pdb file and removes any original heteroatoms. E.g. for comparing different TSs

This script:
1. Reads PDB files from an input directory
2. Performs rigid body alignment of everything in a second pdb file against each pdb file from the original directory
3. Creates merged PDB files with the original structure plus the aligned mobile atoms, 
    less heteroatoms in the original structure
    (If you want to only reposition one cofactor/TS, you'd need to modify this)

Usage:
    python replacer.py input_directory output_directory mobile_pdb "FE1,N1,N4" "FE1,N2,N1"
ORDER MATTERS for the atom lists.

Notes:
This will usually give quite good agreement if more than 3 atoms are defined in both cofactors. RMSD less than 0.05A 
is reasonable for matching porphyrin transition states.
The script will iterate through every pdb in the input directory, using the same mobile pdb, and generate corresponding outputs.
This is not optimized for speed- running thousands of jobs takes tens of minutes on a cpu.
This has various requirements, as listed in the import functions below. I use it in a conda environment that I compiled for ESM (and most protein design environments will have all needed packages).
Enjoy designing enantioselective enzymes!
