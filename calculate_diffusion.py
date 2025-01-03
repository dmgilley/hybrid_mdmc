#!/usr/bin/env python3
#
# Author:
#    Dylan Gilley
#    dgilley@purdue.edu


import argparse, sys, os
import numpy as np
from hybrid_mdmc.data_file_parser import parse_data_file
from hybrid_mdmc.diffusion import Diffusion
from hybrid_mdmc.customargparse import HMDMC_ArgumentParser
from hybrid_mdmc.voxels import Voxels
from hybrid_mdmc.functions import gen_molecules

def main(argv):

    # Use HMDMC_ArgumentParser to parse the command line.
    parser = HMDMC_ArgumentParser(auto=False)
    parser.add_default_args()
    parser.add_argument('-diffusion_species', dest='diffusion_species', type=str, default='all',
                        help='Species for which to calculate diffusion rates. Default is "all".')
    setattr(parser, 'args', parser.parse_args())
    if parser.args.filename_notebook == 'default':
        parser.args.filename_notebook = parser.args.system + '_notebook.xlsx'
    if os.path.isfile(parser.args.filename_notebook):
        parser.read_notebook()
    parser.adjust_default_args()
    args = parser.args
    args.diffusion_species = args.diffusion_species.split()

    # Read in the data_file, diffusion_file, rxndf, and msf files.
    datafile_result = parse_data_file(args.filename_data, unwrap=True, atom_style=args.atom_style)
    atoms, box = datafile_result[0],datafile_result[5]
    masterspecies = parser.get_masterspecies_dict()

    # Create the Voxels object
    voxels_datafile = Voxels(box, args.number_of_voxels)
    atomtypes2moltype = {tuple(sorted([i[2] for i in v['Atoms']])):k for k,v in masterspecies.items()}

    # Create and populate an instance of the "MoleculeList" class.
    molecules = gen_molecules(atoms, atomtypes2moltype, voxels_datafile)

    # Initialize Diffusion and calculate rate for each of the requested species
    diffusion = Diffusion(
        args.prefix,
        args.filename_trajectory,
        atoms,
        molecules,
        voxels_datafile,
        time_conversion=args.lammps_time_units_to_seconds_conversion)
    diffusion.parse_trajectory_file(start=0, end=-1, every=1)
    diffusion.calculate_direct_voxel_transition_rates()

    # If requested, calculate the diffusion rate for each species.
    diffusion_rate = {
        species: np.full((len(voxels_datafile.voxel_IDs), len(voxels_datafile.voxel_IDs)), fill_value=np.inf)
        for species in masterspecies.keys()
    }
    for species in args.diffusion_species:
        diffusion.perform_random_walks(number_of_steps=150,species=species)
        diffusion.calculate_average_first_time_between_positions(species=species)
        diffusion.calculate_diffusion_rates(species=species)
        diffusion_rate[species] = diffusion.diffusion_rates[species]
    
    # Write output
    with open(args.filename_diffusion,'w') as f:
        f.write('\n{}\n\nDiffusionStep {}\n'.format('-'*75,0))
        for k,v in sorted(diffusion_rate.items()):
            f.write('\nDiffusion Rates for {}\n'.format(k))
            for row in v:
                f.write('{}\n'.format(' '.join([str(_) for _ in row])))

    return


if __name__ == '__main__':
    main(sys.argv)