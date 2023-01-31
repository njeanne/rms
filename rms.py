#!/usr/bin/env python3

"""
Created on 09 Dec. 2022
"""

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "1.2.0"

import argparse
import logging
import os
import re
from statistics import mean
import sys

import Bio.PDB
from dna_features_viewer import GraphicFeature, GraphicRecord
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import pytraj as pt
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns


def restricted_float(float_to_inspect):
    """Inspect if a float is between 0.0 and 100.0

    :param float_to_inspect: the float to inspect
    :type float_to_inspect: float
    :raises ArgumentTypeError: is not between 0.0 and 100.0
    :return: the float value if float_to_inspect is between 0.0 and 100.0
    :rtype: float
    """
    x = float(float_to_inspect)
    if x < 0.0 or x > 100.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 100.0]")
    return x


def create_log(path, level):
    """Create the log as a text file and as a stream.

    :param path: the path of the log.
    :type path: str
    :param level: the level og the log.
    :type level: str
    :return: the logging:
    :rtype: logging
    """

    log_level_dict = {"DEBUG": logging.DEBUG,
                      "INFO": logging.INFO,
                      "WARNING": logging.WARNING,
                      "ERROR": logging.ERROR,
                      "CRITICAL": logging.CRITICAL}

    if level is None:
        log_level = log_level_dict["INFO"]
    else:
        log_level = log_level_dict[args.log_level]

    if os.path.exists(path):
        os.remove(path)

    logging.basicConfig(format="%(asctime)s %(levelname)s:\t%(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=log_level,
                        handlers=[logging.FileHandler(path), logging.StreamHandler()])
    return logging


def check_limits(value_arg):
    """Check if the value of the argument is valid.

    :param value_arg: the value of the argument to check.
    :type value_arg: str
    :raises ArgumentTypeError: values not in the fixed limits
    :return: the frames and region of interest limits
    :rtype: list
    """
    lims = []
    pattern = re.compile("(\\d+)-(\\d+)")
    match = pattern.search(value_arg)
    if match:
        min_val = int(match.group(1))
        max_val = int(match.group(2))
        if min_val >= max_val:
            raise argparse.ArgumentTypeError(f"--frames {value_arg} : minimum value {min_val} is > or = to "
                                             f"maximum value {max_val}")
        lims.append(min_val)
        lims.append(max_val)
    else:
        raise argparse.ArgumentTypeError(f"--frames {value_arg} is not a valid format, valid format should be: "
                                         f"--frames <INT>-<INT>")
    return lims


def load_trajectories(trajectory_files, topology_file, ps_frame, frames_lim):
    """
    Load a trajectory and apply a mask if mask argument is set.

    :param trajectory_files: the trajectory file paths.
    :type trajectory_files: list
    :param topology_file: the topology file path.
    :type topology_file: str
    :param ps_frame: the time a frame represents in picoseconds.
    :type ps_frame: float
    :param frames_lim: the frames limits to use for RMSD and RMSF, used to check if this upper limit is not greater
    than the number of frames of the simulation.
    :type frames_lim: list or None
    :return: the loaded trajectory.
    :rtype: pt.Trajectory
    """
    logging.info("Loading trajectory file:")
    logging.info("\tComputing the whole trajectory, please be patient..")
    traj = pt.iterload(trajectory_files, top=topology_file)
    if frames_lim and frames_lim[1] > traj.n_frames:
        raise IndexError(f"Selected upper frame limit for RMS computation ({frames_lim[1]}) from --frames argument "
                         f"is greater than the total frames number ({traj.n_frames}) of the MD trajectory.")
    logging.info(f"\tFrames:\t\t{traj.n_frames}")
    logging.info(f"\tMD duration:\t{traj.n_frames * ps_frame} nanoseconds")
    logging.info(f"\tMolecules:\t{traj.topology.n_mols}")
    logging.info(f"\tResidues:\t{traj.topology.n_residues}")
    logging.info(f"\tAtoms:\t\t{traj.topology.n_atoms}")
    return traj


def extract_pdb(pdb_id, path):
    """
    Read a PDB structure from a file and create a dictionary of the residues numbers with the atoms numbers of this
    residue.

    :param pdb_id: the ID of the PDB file.
    :type pdb_id: str
    :param path: the path to the PDB file.
    :type path: str
    :return: the match between the residue number and the atoms numbers belonging to it.
    :rtype: dict
    """
    logging.info("Extracting the PDB file from the first frame of the trajectory.")
    try:
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)   # suppress PDBConstructionWarning
        struct = pdb_parser.get_structure(pdb_id, path)
        data = {}
        logging.debug("Structure data:")
        for model in struct:
            logging.debug(f"\tmodel: {model.id}")
            for chain in model:
                atom_nb = 1
                res_nb = 1
                logging.debug(f"\t\tchain: {chain.id}")
                for residue in chain:
                    for atom in residue:
                        logging.debug(f"\t\t\tatom {atom_nb} {atom.get_name()}: {residue.get_resname()} {res_nb}")
                        data[atom_nb] = res_nb
                        atom_nb += 1
                    res_nb += 1
    except Exception as ex:
        raise ex

    return data


def rmsf_residues(tmp, data):
    """
    Regroup atoms by residue and compute the average of the RMSF for the residue.

    :param tmp: the RMSF by atom.
    :type tmp: pandas.DataFrame
    :param data: the atom number corresponding to the residue number.
    :type data: dict
    :return: the average RMSF by residue.
    :rtype: pandas.DataFrame
    """
    # get the residues of the RMSF analysis
    residue_rmsf = {"residues": [], "RMSF": []}
    for _, row in tmp.iterrows():
        residue_nb_for_current_atom = data[int(row["atoms"])]
        if residue_nb_for_current_atom not in residue_rmsf["residues"]:
            residue_rmsf["residues"].append(residue_nb_for_current_atom)
    idx_residue_of_interest = 0
    current_residue_nb = residue_rmsf["residues"][idx_residue_of_interest]
    rmsf_for_current_residue = []
    for _, row in tmp.iterrows():
        atom_nb = int(row["atoms"])
        residue_nb_for_current_atom = data[atom_nb]
        if residue_nb_for_current_atom == current_residue_nb:
            rmsf_for_current_residue.append(row["RMSF"])
        else:
            if residue_nb_for_current_atom in residue_rmsf["residues"]:
                if len(rmsf_for_current_residue) > 1:
                    residue_rmsf["RMSF"].append(mean(rmsf_for_current_residue))
                else:  # only one atom, no average computation
                    residue_rmsf["RMSF"].append(rmsf_for_current_residue[0])
            rmsf_for_current_residue = [row["RMSF"]]
            # update the current residue
            idx_residue_of_interest += 1
            current_residue_nb = residue_rmsf["residues"][idx_residue_of_interest]
    # final residue
    if len(rmsf_for_current_residue) > 1:
        residue_rmsf["RMSF"].append(mean(rmsf_for_current_residue))
    else:   # only one atom, no average computation
        residue_rmsf["RMSF"].append(rmsf_for_current_residue[0])
    return pd.DataFrame.from_dict(residue_rmsf)


def plot_rmsd(src, smp, dir_path, fmt, subtitle):
    """
    Plot the RMSD.

    :param src: the data source.
    :type src: pd.Dataframe
    :param smp: the sample name.
    :type smp: str
    :param dir_path: the output directory path.
    :type dir_path: str
    :param fmt: the plot output format.
    :type fmt: str
    :param subtitle: the plot subtitle.
    :type subtitle: str
    :return: the path of the plot.
    :rtype: str
    """
    rms_ax = sns.lineplot(data=src, x="frames", y="RMSD")
    plot = rms_ax.get_figure()
    plt.suptitle(f"Root Mean Square Deviation: {smp.replace('_', ' ')}", fontsize="large", fontweight="bold")
    plt.title(subtitle)
    plt.xlabel("frames", fontweight="bold")
    plt.ylabel("RMSD (\u212B)", fontweight="bold")
    out_path_plot = os.path.join(dir_path, f"RMSD_{smp}.{fmt}")
    plot.savefig(out_path_plot)
    return out_path_plot


def plot_rmsf(src_rmsf, smp, dir_path, fmt, use_dots, subtitle, src_domains=None):
    """
    Plot the RMSF.

    :param src_rmsf: the RMSF data source.
    :type src_rmsf: pd.Dataframe
    :param smp: the sample name.
    :type smp: str
    :param dir_path: the output directory path.
    :type dir_path: str
    :param fmt: the plot output format.
    :type fmt: str
    :param use_dots: if dots should be used to represent the RMSF value of each residue.
    :type use_dots: bool
    :param subtitle: the plot subtitle.
    :type subtitle: str
    :param src_domains: the domains coordinates and info.
    :type src_domains: Pandas.Dataframe
    :return: the path of the plot.
    :rtype: str
    """
    if src_domains is not None:
        fig, axs = plt.subplots(2, 1, layout="constrained", height_ratios=[10, 1])
        # axes 0
        if use_dots:
            sns.lineplot(data=src_rmsf, x="residues", y="RMSF", ax=axs[0], marker="o")
        else:
            sns.lineplot(data=src_rmsf, x="residues", y="RMSF", ax=axs[0])
        axs[0].set_ylabel("RMSF (\u212B)", fontweight="bold")
        axs[0].set_xlabel("residues", fontweight="bold")
        axs[0].set_title(subtitle)
        # axes 1
        features = []
        row = None
        for _, row in src_domains.iterrows():
            features.append(GraphicFeature(start=row["start"], end=row["end"], strand=+1, color=row["color"],
                                           label=row["domain"]))
        # set the last residue for the X axes 0 superior limit matches with the domains representation
        axs[0].set_xlim(0, row["end"] + 1)

        record = GraphicRecord(sequence_length=row["end"] + 1, features=features, plots_indexing="genbank")
        ax_domains, _ = record.plot(ax=axs[1])
    else:
        fig, axs = plt.subplots(1, 1, layout="constrained")
        sns.lineplot(data=src_rmsf, x="residues", y="RMSF")
        axs.set_ylabel("RMSF (\u212B)", fontweight="bold")
        axs.set_xlabel("residues", fontweight="bold")
        axs.set_xlim(0, max(src_rmsf["residues"] + 1))
        axs.set_title(subtitle)
    fig.suptitle(f"Root Mean Square Fluctuation: {smp.replace('_', ' ')}", fontsize="large", fontweight="bold")
    fig.tight_layout()
    out_path_plot = os.path.join(dir_path, f"RMSF_{smp}.{fmt}")
    fig.savefig(out_path_plot)
    return out_path_plot


def rms(rms_type, traj, out_dir, out_basename, format_output, use_dots_for_rmsf, ns_frame=None, frames_lim=None,
        mask=None, atom_from_res=None, domains=None):
    """
    Compute the Root Mean Square Deviation or the Root Mean Square Fluctuation and create the plot.

    :param rms_type: the type of analysis, RMSD or RMSF.
    :type rms_type: str
    :param traj: the trajectory.
    :type traj: pt.Trajectory
    :param out_dir: the output directory path
    :type out_dir: str
    :param out_basename: the plot and CSV basename.
    :type out_basename: str
    :param format_output: the output format for the plots.
    :type format_output: str
    :param use_dots_for_rmsf: if dots should be used to represent the RMSF value of each residue.
    :type use_dots_for_rmsf: bool
    :param ns_frame: the time a frame represents in nanoseconds.
    :type ns_frame: float
    :param frames_lim: the frames limits.
    :type frames_lim: list
    :param mask: the applied mask.
    :type mask: str
    :param atom_from_res: the atom number corresponding to a residue number.
    :type atom_from_res: dict
    :param domains: the domains coordinates and info.
    :type: Pandas.Dataframe
    :raises ValueError: unknown RMS type
    """
    log_txt = f"{rms_type} computation"
    if mask:
        log_txt = f"{log_txt}, with Mask {mask}"
    if frames_lim:
        range_frames = [x for x in range(frames_lim[0], frames_lim[1])]
        log_txt = f"{log_txt}, using frames {frames_lim[0]} to {frames_lim[1]} and frame 0 as reference."
    else:
        range_frames = [x for x in range(traj.n_frames)]
    logging.info(f"{log_txt}:")

    path_csv = f"{os.path.join(out_dir, f'{rms_type}_{out_basename}')}.csv"

    subtitle_plot = None
    if mask:
        subtitle_plot = f"Applied mask: {mask}"
    if frames_lim:
        sep = ""
        if subtitle_plot:
            sep = "   "
        subtitle_plot = f"{subtitle_plot}{sep}Frames used: {frames_lim[0]}-{frames_lim[1]} on {trajectory.n_frames} " \
                        f"frames"

    if rms_type == "RMSD":
        rmsd_traj = pt.rmsd(traj, mask=mask, ref=0, frame_indices=range_frames)
        time_rmsd = [x * ns_frame for x in range_frames]
        x_axis_name = "time (ns)"
        source = pd.DataFrame({"frames": range_frames, x_axis_name: time_rmsd, f"{rms_type}": rmsd_traj})
        plot_path = plot_rmsd(source, out_basename, out_dir, format_output, subtitle_plot)
    elif rms_type == "RMSF":
        rmsf_traj = pt.rmsf(traj, mask=mask)
        tmp_source = pd.DataFrame({"atoms": rmsf_traj.T[0], f"{rms_type}": rmsf_traj.T[1]})
        source = rmsf_residues(tmp_source, atom_from_res)
        subtitle_plot = f"{subtitle_plot}\nAverage RMSF of the atoms by residues"
        plot_path = plot_rmsf(source, out_basename, out_dir, format_output, use_dots_for_rmsf, subtitle_plot, domains)
    else:
        raise ValueError(f"{rms_type} is not a valid case, only \"RMSD\" or \"RMSF\" are allowed.")
    source.to_csv(path_csv, index=False)
    logging.info(f"\tdata saved: {path_csv}")
    logging.info(f"\tplot {rms_type} saved: {plot_path}")


if __name__ == "__main__":
    descr = f"""
    {os.path.basename(__file__)} v. {__version__}

    Created by {__author__}.
    Contact: {__email__}
    {__copyright__}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied.

    From a molecular dynamics trajectory file perform trajectory analysis. The script computes the Root Mean Square 
    Deviation (RMSD) and the Root Mean Square Fluctuation (RMSF).
    WARNING: the mask selection is only used to compute the RMSD and RMSF plots not for loading the trajectory because 
    if the mask defined the backbone, no hydrogen bond will be find.
    See: https://amber-md.github.io/pytraj/latest/atom_mask_selection.html#examples-atom-mask-selection-for-trajectory
    """
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", required=True, type=str, help="the path to the output directory.")
    parser.add_argument("-t", "--topology", required=True, type=str,
                        help="the path to the molecular dynamics topology file.")
    parser.add_argument("-p", "--ps-by-frame", required=True, type=float,
                        help="the elapsed time in picoseconds for each frame as set in the MD configuration file.")
    parser.add_argument("-x", "--frames", required=False, type=str,
                        help="the frames to use for the RMSD and the RMSF, the format must be two integers separated "
                             "by an hyphen, i.e to load the trajectory from the frame 500 to 2000: --frames 500-2000")
    parser.add_argument("-m", "--mask", required=False, type=str, default="", help="the residues mask selection.")
    parser.add_argument("-d", "--domains", required=False, type=str, default="",
                        help="the path to the CSV domains file. A comma separated file, the first column is the "
                             "annotation name, the 2nd is the residue start coordinate, the 3rd is the residue end "
                             "coordinate, the last one is the color to apply in hexadecimal format. The coordinate are "
                             "1-indexed.")
    parser.add_argument("--dots-for-residues", required=False, action="store_true",
                        help="use dots on the RMSF plot for each residue, useful when a mask is used and not all the "
                             "protein residues are used.")
    parser.add_argument("-f", "--format", required=False, default="svg",
                        choices=["eps", "jpg", "jpeg", "pdf", "pgf", "png", "ps", "raw", "svg", "svgz", "tif", "tiff"],
                        help="the output plots format: 'eps': 'Encapsulated Postscript', "
                             "'jpg': 'Joint Photographic Experts Group', 'jpeg': 'Joint Photographic Experts Group', "
                             "'pdf': 'Portable Document Format', 'pgf': 'PGF code for LaTeX', "
                             "'png': 'Portable Network Graphics', 'ps': 'Postscript', 'raw': 'Raw RGBA bitmap', "
                             "'rgba': 'Raw RGBA bitmap', 'svg': 'Scalable Vector Graphics', "
                             "'svgz': 'Scalable Vector Graphics', 'tif': 'Tagged Image File Format', "
                             "'tiff': 'Tagged Image File Format'. Default is 'svg'.")
    parser.add_argument("--remove-pdb", required=False, action="store_true",
                        help="if the PDB file extracted from the trajectory should be removed.")
    parser.add_argument("-l", "--log", required=False, type=str,
                        help="the path for the log file. If this option is skipped, the log file is created in the "
                             "output directory.")
    parser.add_argument("--log-level", required=False, type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="set the log level. If the option is skipped, log level is INFO.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("inputs", nargs="+", type=str,
                        help="the paths to the molecular dynamics trajectory files (*.nc).")
    args = parser.parse_args()

    # create output directory if necessary
    os.makedirs(args.out, exist_ok=True)
    # create the logger
    if args.log:
        log_path = args.log
    else:
        log_path = os.path.join(args.out, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
    create_log(log_path, args.log_level)

    logging.info(f"version: {__version__}")
    logging.info(f"CMD: {' '.join(sys.argv)}")
    try:
        frames_limits = check_limits(args.frames)
    except argparse.ArgumentTypeError as exc:
        logging.error(exc)
        sys.exit(1)

    domains_data = None
    if args.domains:
        try:
            domains_data = pd.read_csv(args.domains)
        except FileNotFoundError as exc:
            logging.error(exc)
            sys.exit(1)

    # set the seaborn plots theme and size
    sns.set_theme()
    rcParams["figure.figsize"] = 15, 12

    # load the trajectory
    try:
        trajectory = load_trajectories(args.inputs, args.topology, args.ps_by_frame, frames_limits)
    except RuntimeError as exc:
        logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({args.input}) files exists",
                      exc_info=True)
        sys.exit(1)
    except ValueError as exc:
        logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({args.input}) files exists.",
                      exc_info=True)
        sys.exit(1)
    except IndexError as exc:
        logging.error(exc, exc_info=True)
        sys.exit(1)

    # extracting .pdb file from the trajectory
    basename = os.path.splitext(os.path.basename(args.input))[0]
    pdb_path = os.path.join(args.out, f"{basename}.pdb")
    pt.write_traj(pdb_path, traj=trajectory, overwrite=True, frame_indices=[0, 1])
    # load the .pdb file
    atom_res = extract_pdb(basename, pdb_path)

    try:
        # compute RMSD and create the plot
        rms("RMSD", trajectory, args.out, basename, args.format, args.dots_for_residues, args.ps_by_frame,
            frames_limits, args.mask)
        # compute RMSF and create the plot
        rms("RMSF", trajectory, args.out, basename, args.format, args.dots_for_residues, args.ps_by_frame,
            frames_limits, args.mask, atom_res, domains_data)
    except ValueError as exc:
        logging.error(exc, exc_info=True)
        sys.exit(1)

    if args.remove_pdb:
        logging.info("Extracted PDB file removed.")
        os.remove(pdb_path)
