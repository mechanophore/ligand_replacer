#!/usr/bin/env python3
"""
Cofactor Alignment and Merging Script

Developed by Maggie Horst
Claude 4 was used during scripting

This script:
1. Reads PDB files from an input directory
2. Performs rigid body alignment of a cofactor in a mobile_pdb file each occurance of a 
    cofactor in an input pdb in the input directory
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
Enjoy designing selective enzymes for directed evolution!
"""

import numpy as np
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path


class FastPDBAligner:
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        self.io = PDBIO()
        # Caching for mobile structure
        self._mobile_structure_cache = None
        self._mobile_atoms_cache = None
        self._mobile_pdb_path = None

    def read_pdb(self, pdb_file):
        try:
            return self.parser.get_structure('structure', pdb_file)
        except Exception as e:
            print(f"Error reading PDB file {pdb_file}: {e}")
            return None

    def cache_mobile_structure(self, mobile_pdb, mobile_atom_names):
        if self._mobile_pdb_path != mobile_pdb or self._mobile_structure_cache is None:
            print(f"Caching mobile structure: {mobile_pdb}")
            self._mobile_structure_cache = self.read_pdb(mobile_pdb)
            self._mobile_atoms_cache = self.find_alignment_atoms(
                self._mobile_structure_cache, mobile_atom_names
            )
            self._mobile_pdb_path = mobile_pdb

    def find_alignment_atoms(self, structure, target_atoms):
        if structure is None or not target_atoms:
            return {}
        
        alignment_atoms = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.name in target_atoms:
                            alignment_atoms[atom.name] = atom.coord.copy()
        return alignment_atoms

    def rigid_transform_3d_vectorized(self, coords, translation, rotation_vector):
        if coords.size == 0:
            return np.array([])
        
        rotation = Rotation.from_rotvec(rotation_vector)
        rotation_matrix = rotation.as_matrix()

        transformed = np.dot(coords, rotation_matrix.T) + translation
        return transformed

    def calculate_rmsd(self, coords1, coords2):
        if coords1.size == 0 or coords2.size == 0 or coords1.shape != coords2.shape:
            return np.inf

        diff = coords1 - coords2
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    def objective_function(self, params, mobile_coords, target_coords):
        translation = params[:3]
        rotation_vector = params[3:6]

        transformed_coords = self.rigid_transform_3d_vectorized(mobile_coords, translation, rotation_vector)
        rmsd = self.calculate_rmsd(transformed_coords, target_coords)
        return rmsd

    def align_structures_fast(self, original_structure, original_atom_names, mobile_atom_names):
        if original_structure is None or self._mobile_structure_cache is None:
            print("Error: Cannot perform alignment due to missing structure(s).")
            return np.zeros(3), np.zeros(3)

        if not original_atom_names or not mobile_atom_names:
            print("Error: Cannot perform alignment due to missing atom names.")
            return np.zeros(3), np.zeros(3)

        # Find atoms in original structure
        original_atoms_dict = self.find_alignment_atoms(original_structure, original_atom_names)
        
        # Use cached mobile atoms
        mobile_atoms_dict = self._mobile_atoms_cache

        # Check if all required atoms are present
        missing_original = set(original_atom_names) - set(original_atoms_dict.keys())
        if missing_original:
            print(f"Error: Missing required alignment atoms in original structure: {missing_original}")
            return np.zeros(3), np.zeros(3)

        missing_mobile = set(mobile_atom_names) - set(mobile_atoms_dict.keys())
        if missing_mobile:
            print(f"Error: Missing required alignment atoms in mobile structure: {missing_mobile}")
            return np.zeros(3), np.zeros(3)

        # Ensure the number of atoms to align is the same and sufficient
        if len(original_atom_names) != len(mobile_atom_names) or len(original_atom_names) < 3:
            print(f"Error: Number of original atoms ({len(original_atom_names)}) and mobile atoms ({len(mobile_atom_names)}) must be the same and at least 3 for alignment.")
            return np.zeros(3), np.zeros(3)

        print(f"Aligning original atoms {original_atom_names} to mobile atoms {mobile_atom_names}")

        # Prepare coordinate arrays based on the order of atom names provided
        original_coords = np.array([original_atoms_dict[name] for name in original_atom_names])
        mobile_coords = np.array([mobile_atoms_dict[name] for name in mobile_atom_names])

        # Initial parameters (no transformation)
        initial_params = np.zeros(6)  # [tx, ty, tz, rx, ry, rz]

        # Optimize
        result = minimize(
            self.objective_function,
            initial_params,
            args=(mobile_coords, original_coords),
            method='BFGS'
        )

        if not result.success:
            print("Warning: Optimization did not converge successfully")

        optimal_translation = result.x[:3]
        optimal_rotation_vector = result.x[3:6]
        final_rmsd = result.fun

        print(f"Final RMSD: {final_rmsd:.3f} Å")
        print(f"Translation: [{optimal_translation[0]:.3f}, {optimal_translation[1]:.3f}, {optimal_translation[2]:.3f}]")
        print(f"Rotation (rad): [{optimal_rotation_vector[0]:.3f}, {optimal_rotation_vector[1]:.3f}, {optimal_rotation_vector[2]:.3f}]")

        return optimal_translation, optimal_rotation_vector

    def transform_structure_vectorized(self, structure, translation, rotation_vector):
        """Apply transformation to all atoms in structure using vectorized operations"""
        if structure is None:
            return

        rotation = Rotation.from_rotvec(rotation_vector)
        rotation_matrix = rotation.as_matrix()

        # Collect all coordinates and atom references
        all_coords = []
        atom_refs = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        all_coords.append(atom.coord.copy())
                        atom_refs.append(atom)
                        if not all_coords:
                            return

        # Convert to numpy array for vectorized operations
        coords_array = np.array(all_coords)
        
        # Vectorized transformation: apply rotation then translation
        transformed_coords = np.dot(coords_array, rotation_matrix.T) + translation

        # Update atom coordinates
        for i, atom in enumerate(atom_refs):
            atom.coord = transformed_coords[i]

    def create_merged_structure(self, original_structure, transformed_structure, output_path):
        """Create new PDB with original structure (minus HETATOMs) plus transformed structure"""
        if original_structure is None or transformed_structure is None:
            print("Error: Cannot create merged structure due to missing input structure(s).")
            return

        # Create new structure
        new_structure = Structure.Structure('merged')
        new_model = Model.Model(0)
        new_structure.add(new_model)

        # Copy non-HETATOM residues from original structure, excluding chain 'X'
        for model in original_structure:
            for chain in model:
                # Skip chains with ID 'X' from the original structure
                if chain.id == 'X':
                    continue

                # Ensure chain ID is valid for PDB format
                chain_id = chain.id if len(chain.id) <= 4 and ' ' not in chain.id else f"A_{chain.id}"[:4]
                new_chain = Chain.Chain(chain_id)
                new_model.add(new_chain)

                for residue in chain:
                    if residue.id[0] == ' ':  # Only ATOM records (not HETATOM)
                        # Create new residue using deepcopy to avoid modifying original
                        new_residue = deepcopy(residue)
                        new_chain.add(new_residue)

        # Add all atoms from transformed structure as a new chain
        transformed_chain_id = "X"
        existing_chain_ids = [chain.id for chain in new_model.get_chains()]
        if transformed_chain_id in existing_chain_ids:
            i = 1
            while f"{transformed_chain_id}{i}" in existing_chain_ids:
                i += 1
            transformed_chain_id = f"{transformed_chain_id}{i}"

        new_transformed_chain = Chain.Chain(transformed_chain_id)
        new_model.add(new_transformed_chain)
        residue_id_counter = 1

        for model in transformed_structure:
            for chain in model:
                for residue in chain:
                    # Create new residue with a unique ID using deepcopy
                    new_residue = deepcopy(residue)
                    new_residue.id = (' ', residue_id_counter, ' ')
                    new_transformed_chain.add(new_residue)
                    residue_id_counter += 1

        # Write the merged structure
        try:
            self.io.set_structure(new_structure)
            self.io.save(output_path)
            print(f"Merged structure saved to: {output_path}")
        except Exception as e:
            print(f"Error saving merged PDB file {output_path}: {e}")

    def process_pdbs_fast(self, original_pdb, mobile_pdb, output_path, original_target_atoms, mobile_target_atoms):
        """Optimized processing function with caching"""
        # Cache mobile structure and atoms (only done once or when mobile_pdb changes)
        self.cache_mobile_structure(mobile_pdb, mobile_target_atoms)

        print(f"Reading original PDB: {original_pdb}")
        original_structure = self.read_pdb(original_pdb)

        if original_structure is None:
            print("Error: Could not load original PDB file. Aborting.")
            return

        if self._mobile_structure_cache is None:
            print("Error: Could not load mobile PDB file. Aborting.")
            return

        print("Performing rigid body alignment...")
        translation, rotation = self.align_structures_fast(
            original_structure, original_target_atoms, mobile_target_atoms
        )

        # Check if alignment was successful
        if np.all(translation == 0) and np.all(rotation == 0) and len(original_target_atoms) < 3:
            print("Alignment failed or not enough common atoms found. Skipping transformation and merging.")
            return

        mobile_structure_copy = deepcopy(self._mobile_structure_cache)

        print("Applying transformation to mobile structure...")
        self.transform_structure_vectorized(mobile_structure_copy, translation, rotation)

        print("Creating merged structure...")
        self.create_merged_structure(original_structure, mobile_structure_copy, output_path)


def parse_atom_list(atom_string):
    """Parse comma-separated atom names"""
    return [atom.strip() for atom in atom_string.split(',')]


def quick_structure_check(pdb_file, required_atoms):
    """Quick check if structure has required atoms without full parsing"""
    try:
        with open(pdb_file, 'r') as f:
            atom_names = set()
            for line in f:
                if line.startswith(('ATOM  ', 'HETATM')):
                    atom_name = line[12:16].strip()
                    atom_names.add(atom_name)
                    # Early termination if all required atoms found
                    if all(atom in atom_names for atom in required_atoms):
                        return True
            return False
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Align and merge PDB structures (Optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ts_mover.py /path/to/pdbs /path/to/output reference.pdb "CA,CB,CG" "CA,CB,CG"
        """
    )
    
    parser.add_argument('input_directory', 
                       help='Directory containing PDB files to process')
    parser.add_argument('output_directory', 
                       help='Directory to save merged PDB files')
    parser.add_argument('mobile_pdb', 
                       help='Mobile PDB file to align to each input PDB')
    parser.add_argument('original_target_atoms', 
                       help='Comma-separated list of atom names in original structures (e.g., "FE1,N1,N4,N3,N2")')
    parser.add_argument('mobile_target_atoms', 
                       help='Comma-separated list of atom names in mobile structure (e.g., "FE1,N2,N1,N4,N3")')
    
    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_directory)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        sys.exit(1)

    # Validate mobile PDB file
    mobile_pdb = Path(args.mobile_pdb)
    if not mobile_pdb.exists():
        print(f"Error: Mobile PDB file '{mobile_pdb}' does not exist.")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Parse atom lists
    try:
        original_atoms = parse_atom_list(args.original_target_atoms)
        mobile_atoms = parse_atom_list(args.mobile_target_atoms)
    except Exception as e:
        print(f"Error parsing atom lists: {e}")
        sys.exit(1)

    if len(original_atoms) != len(mobile_atoms):
        print(f"Error: Number of original atoms ({len(original_atoms)}) must match number of mobile atoms ({len(mobile_atoms)})")
        sys.exit(1)

    if len(original_atoms) < 3:
        print(f"Error: At least 3 atoms are required for alignment, got {len(original_atoms)}")
        sys.exit(1)

    print(f"Original target atoms: {original_atoms}")
    print(f"Mobile target atoms: {mobile_atoms}")

    # Find all PDB files in input directory
    pdb_files = list(input_dir.glob("*.pdb"))
    if not pdb_files:
        print(f"Warning: No PDB files found in '{input_dir}'")
        sys.exit(1)

    print(f"Found {len(pdb_files)} PDB files")

    # Pre-filter PDB files for speed
    print("Pre-filtering PDB files for required atoms...")
    valid_pdb_files = []
    for pdb_file in pdb_files:
        if quick_structure_check(pdb_file, original_atoms):
            valid_pdb_files.append(pdb_file)
        else:
            print(f"⚠ Skipping {pdb_file.name}: missing required atoms")

    if not valid_pdb_files:
        print("Error: No valid PDB files found with required atoms")
        sys.exit(1)

    print(f"Processing {len(valid_pdb_files)} valid PDB files")

    # Initialize optimized aligner
    aligner = FastPDBAligner()

    # Process each valid PDB file
    successful = 0
    failed = 0

    for pdb_file in valid_pdb_files:
        output_path = output_dir / pdb_file.name
        
        print(f"\n{'='*60}")
        print(f"Processing {pdb_file.name}...")
        print(f"{'='*60}")
        
        try:
            aligner.process_pdbs_fast(
                str(pdb_file),
                str(mobile_pdb),
                str(output_path),
                original_atoms,
                mobile_atoms
            )
            successful += 1
            print(f"✓ Successfully processed {pdb_file.name}")
            
        except Exception as e:
            print(f"✗ Failed to process {pdb_file.name}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Skipped (missing atoms): {len(pdb_files) - len(valid_pdb_files)} files")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()