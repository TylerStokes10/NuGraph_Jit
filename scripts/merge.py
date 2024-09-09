#!/usr/bin/env python
import argparse
import os
import glob
import tqdm
import h5py

from nugraph.data import H5DataModule

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-pattern', type=str, required=True,
                        help='Glob pattern for input HDF5 files')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='Output HDF5 file name')
    return parser.parse_args()

def merge(args):
    # open final output file
    with h5py.File(args.output_file, 'w', libver='latest') as fout:
        data_out = fout.create_group('dataset')

        # loop over each input file to merge it in
        for fname in tqdm.tqdm(glob.glob(args.input_pattern)):
            try:
                with h5py.File(fname, 'r') as fin:

                    # loop over keys in input file
                    for key in fin.keys():
                        data_in = fin[key]
                        # if it's the dataset group, loop over graphs and attempt to copy
                        if key == 'dataset':
                            for graph in data_in.keys():
                                if graph not in data_out:
                                    fin.copy(data_in[graph], data_out, graph)
                                else:
                                    raise RuntimeError(f"Conflict detected for {graph}. Skipping file {fname}.")

                        # otherwise it's metadata, so just copy it directly
                        else:
                            if key not in fout:
                                fin.copy(data_in, fout, key)
                            else:
                                raise RuntimeError(f"Conflict detected for {key}. Skipping file {fname}.")

                # delete temporary file once it's been merged
                os.remove(fname)

            except RuntimeError as e:
                print(e)
                continue  # Skip this file and continue with the next one

    # prepare dataset
    H5DataModule.generate_samples(args.output_file)
    H5DataModule.generate_norm(args.output_file, 64)

if __name__ == '__main__':
    args = configure()
    merge(args)

