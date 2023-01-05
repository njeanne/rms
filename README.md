# rms

Compute the [Root Mean Square Deviation](https://amberhub.chpc.utah.edu/amber-hub/start-here-rmsd-analysis-in-cpptraj/) (RMSD) and the [Root Mean Square Factors](https://amberhub.chpc.utah.edu/atomicfluct-rmsf/) (RMSF) plots from a trajectory file.

## Conda environment

A [conda](https://docs.conda.io/projects/conda/en/latest/index.html) YAML environment file is provided: `rms_env.yml`. The file contains all the dependencies to run the script.
The conda environment is created using the command:
```shell script
# create the environment
conda env create -f rms_env.yml

# activate the environment
conda activate rms
```

## Commands

The usage of the script after activating the conda environment can be displayed with:

```shell script
./rms.py -h
```

The analysis can be run with the following command on the test data provided in the repository:
```shell script
./rms.py --out results --ps-by-frame 0.002 --mask @CA,C,O,N --format svg --remove-pdb --topology data/MD_data_test.parm data/MD_data_test.nc
```
The `--ps-by-frame` argument is the time elapsed at each frame of the molecular dynamics simulation.
The `--mask` argument is the mask to apply on the residues to compute the RMSs as described in [AMBER documentation](https://amber-md.github.io/pytraj/latest/atom_mask_selection.html#examples-atom-mask-selection-for-trajectory).

If you want to add the domains annotation of the protein, you can use the `--domains` argument which value is the path to a comma separated file with the domains coordinates as the one provided in `data/MD_data_test_domains.csv`.

## Results

The script produces a RMSD plot:

![RMSD plot](doc/_static/RMSD.svg)

the RMSF plot **without** the `--domains` option:

![RMSF plot](doc/_static/RMSF_without_domains.svg)

or the RMSF plot **with** the `--domains` option:

![RMSF plot](doc/_static/RMSF_with_domains.svg)

and the associated CSV files.# rms
