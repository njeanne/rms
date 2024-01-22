#!/usr/bin/env python3

"""
Created on 09 Dec. 2022
"""

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "1.4.1"

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
import numpy
import pandas as pd
import pytraj as pt
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
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
        log_level = log_level_dict[level]

    if os.path.exists(path):
        os.remove(path)

    logging.basicConfig(format="%(asctime)s %(levelname)s:\t%(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=log_level,
                        handlers=[logging.FileHandler(path), logging.StreamHandler()])
    return logging


def load_trajectories(trajectory_files, topology_file, info):
    """
    Load a trajectory and apply a mask if mask argument is set.

    :param trajectory_files: the trajectory file paths.
    :type trajectory_files: list
    :param topology_file: the topology file path.
    :type topology_file: str
    :param info: the molecular dynamics information as free text.
    :type info: str
    :return: the loaded trajectory.
    :rtype: pt.Trajectory
    """
    logging.info("Loading trajectory file:")
    logging.info("\tComputing the whole trajectory, please be patient..")
    traj = pt.iterload(trajectory_files, top=topology_file)
    logging.info(f"\tInformation:\t{info}")
    logging.info(f"\tMolecules:\t{traj.topology.n_mols}")
    logging.info(f"\tResidues:\t{traj.topology.n_residues}")
    logging.info(f"\tAtoms:\t\t{traj.topology.n_atoms}")
    logging.info(f"\tFrames:\t\t{traj.n_frames}")
    return traj


def check_limits(limit_arg, reference_frame, traj):
    """Check if the limits are valid.

    :param limit_arg: the value of the limits' argument to check.
    :type limit_arg: str or None
    :param reference_frame: the reference frame for the RMSD and the RMSF.
    :type reference_frame: int
    :param traj: the trajectory.
    :type traj: pt.Trajectory
    :raises ArgumentTypeError: total number of frames not in the frames' limits or invalid format of the --frames
    argument.
    :return: the frames' limits.
    :rtype: list
    """
    min_val = 0
    max_val = traj.n_frames
    if limit_arg:
        pattern = re.compile("(\\d*):(\\d*)")
        match = pattern.search(limit_arg)
        if match:
            if match.group(1) and match.group(2):
                min_val = int(match.group(1))
                max_val = int(match.group(2))
                if min_val >= max_val:
                    raise argparse.ArgumentTypeError(f"--frames {limit_arg} : minimum value {min_val} is > or = to "
                                                     f"maximum value {max_val}")
            elif match.group(1):
                min_val = int(match.group(1))
            elif match.group(2):
                max_val = int(match.group(2))
            if reference_frame:
                if reference_frame < min_val:
                    raise argparse.ArgumentTypeError(f"--ref-frame: {reference_frame} is below first frame \"{min_val}\".")
                if reference_frame > max_val:
                    raise argparse.ArgumentTypeError(f"--ref-frame: {reference_frame} is higher than last frame "
                                                     f"\"{max_val}\".")
        else:
            raise argparse.ArgumentTypeError(f"--frames {limit_arg} is not a valid format, valid format should be: "
                                             f"--frames <INT>:<INT> or --frames :<INT> or --frames <INT>:")
    # check if the limits fit with the trajectory
    if max_val > traj.n_frames:
        raise IndexError(f"Selected upper frame limit for RMS computation ({max_val}) from --frames argument is "
                         f"greater than the total frames number ({traj.n_frames}) of the MD trajectory.")
    return [min_val, max_val]


def link_atoms_to_residue_from_pdb(pdb_id, path):
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
    logging.info("Extracting the atoms by residue from the PDB file generated with the first frame of the whole "
                 "trajectory.")
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


def get_reference_frame(traj, mask, frames_lim, step, path_sampled):
    """
    Get the most common frame using a clustering on the trajectory frame.

    :param traj: the trajectory.
    :type traj: pt.Trajectory
    :param mask: the applied mask.
    :type mask: str
    :param frames_lim: the frames' limits to use for RMSD and RMSF, used to check if this upper limit is not greater
    than the simulation's number of frames.
    :type frames_lim: list
    :param step: the step to apply for the frames' selection.
    :type: int
    :param path_sampled: the path of the output sampled frame as a PDB structure file.
    :type path_sampled: str
    :return: the most represented cluster frame number.
    :rtype: int
    """
    logging.info("Computing the trajectory clustering:")
    if step != 1:
        # sample the trajectory
        logging.info(f"\tSelecting every {step} frames from {frames_lim[0]} to {frames_lim[1]}.")
        traj_to_clusterize = traj(frames_lim[0], frames_lim[1], step)
    else:
        logging.info(f"\tSelecting {frames_lim[0]} to {frames_lim[1]} frames.")
        traj_to_clusterize = traj(frames_lim[0], frames_lim[1])
    logging.info(f"\tClustering performed on {traj_to_clusterize.n_frames} frames with the mask '{mask}':")
    clusters_data = pt.cluster.kmeans(traj_to_clusterize, mask=mask, n_clusters=5)
    clusters_frames_counts = numpy.bincount(clusters_data.cluster_index)
    max_idx = clusters_frames_counts.argmax()
    most_frequent_cluster_centroid_frame = int(clusters_data.centroids[max_idx])
    ref_frame_index_on_trajectory = frames_lim[0] + (most_frequent_cluster_centroid_frame - 1) * step
    logging.info(f"\t\tCluster {max_idx} with the centroid frame {most_frequent_cluster_centroid_frame} is the most "
                 f"representative cluster on the {traj_to_clusterize.n_frames} sampled frames: "
                 f"{clusters_frames_counts[max_idx]}/{len(clusters_data.cluster_index)} "
                 f"occurrences")
    logging.info(f"\t\tWhole trajectory reference frame: {ref_frame_index_on_trajectory} (sampled reference frame "
                 f"{most_frequent_cluster_centroid_frame}, index {most_frequent_cluster_centroid_frame - 1})")
    # write the reference frame as a PDB file
    pt.write_traj(path_sampled, traj=traj, overwrite=True, frame_indices=[most_frequent_cluster_centroid_frame])
    logging.info(f"\t\tSampled frame from the clustering written to PDB format: {os.path.abspath(path_sampled)}")
    logging.info("\t\tClusters size:")
    # display information on the cluster size
    for idx in range(len(clusters_frames_counts)):
        logging.info(f"\t\t\t- cluster {idx}: {clusters_frames_counts[idx]}/{len(clusters_data.cluster_index)} frames")
    return ref_frame_index_on_trajectory


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


def plot_rmsd_line(src, smp, dir_path, fmt, subtitle):
    """
    Create the RMSD line plot.

    :param src: the data source.
    :type src: pd.Dataframe
    :param smp: the sample name.
    :type smp: str
    :param dir_path: the output directory path.
    :type dir_path: str
    :param fmt: the plot output format.
    :type fmt: str
    :param subtitle: the plot's subtitle.
    :type subtitle: str
    :return: the path of the plot.
    :rtype: str
    """
    rms_line_ax = sns.lineplot(data=src, x="frames", y="RMSD")
    plot = rms_line_ax.get_figure()
    plt.suptitle(f"Root Mean Square Deviation: {smp.replace('_', ' ')}", fontsize="large", fontweight="bold")
    plt.title(subtitle)
    plt.xlabel("frames", fontweight="bold")
    plt.ylabel("RMSD (\u212B)", fontweight="bold")
    out_path_plot = os.path.join(dir_path, f"RMSD_{smp}")
    out_path_plot = f"{out_path_plot}.{fmt}"
    plot.savefig(out_path_plot)
    return out_path_plot


def plot_rmsd_histogram(src, smp, dir_path, fmt, subtitle):
    """
    Create the RMSD histogram.

    :param src: the data source.
    :type src: pd.Dataframe
    :param smp: the sample name.
    :type smp: str
    :param dir_path: the output directory path.
    :type dir_path: str
    :param fmt: the plot output format.
    :type fmt: str
    :param subtitle: the plot's subtitle.
    :type subtitle: str
    :return: the path of the plot.
    :rtype: str
    """
    # clear the previous RMSD line plot
    plt.clf()
    # create the histogram
    rms_histogram_ax = sns.histplot(data=src, x="RMSD", stat="density", kde=True)
    histogram = rms_histogram_ax.get_figure()
    plt.suptitle(f"Root Mean Square Deviation histogram: {smp.replace('_', ' ')}", fontsize="large", fontweight="bold")
    plt.title(subtitle)
    plt.xlabel("RMSD (\u212B)", fontweight="bold")
    plt.ylabel("Density", fontweight="bold")
    out_path_plot = os.path.join(dir_path, f"RMSD_histogram_{smp}")
    out_path_plot = f"{out_path_plot}.{fmt}"
    histogram.savefig(out_path_plot)
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
    :param subtitle: the plot's subtitle.
    :type subtitle: str
    :param src_domains: the domains' coordinates and info.
    :type src_domains: Pandas.Dataframe
    :return: the path of the plot.
    :rtype: str
    """
    if src_domains is not None:
        # create the domain's map and the RMSF plot
        fig, axs = plt.subplots(2, 1, layout="constrained", height_ratios=[10, 1])
        # RMSF plot
        if use_dots:
            sns.lineplot(data=src_rmsf, x="residues", y="RMSF", ax=axs[0], marker="o")
        else:
            sns.lineplot(data=src_rmsf, x="residues", y="RMSF", ax=axs[0])
        axs[0].axhline(src_rmsf["RMSF"].median(), color="red", lw=2)
        axs[0].set_ylabel("RMSF (\u212B)", fontweight="bold")
        axs[0].set_xlabel("residues", fontweight="bold")
        axs[0].set_title(subtitle)
        axs[0].legend(handles=[Line2D([0], [0], color="red", lw=2, label='median RMSF')], loc='best')
        # Domains plot
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
        # create the RMSF plot only
        fig, axs = plt.subplots(1, 1, layout="constrained")
        sns.lineplot(data=src_rmsf, x="residues", y="RMSF")
        axs.axhline(src_rmsf["RMSF"].median(), color="red", lw=2)
        axs.set_ylabel("RMSF (\u212B)", fontweight="bold")
        axs.set_xlabel("residues", fontweight="bold")
        axs.set_xlim(min(src_rmsf["residues"]), max(src_rmsf["residues"] + 1))
        axs.set_title(subtitle)
        axs.legend(handles=[Line2D([0], [0], color="red", lw=2, label='median RMSF')], loc='best')
    fig.suptitle(f"Root Mean Square Fluctuation: {smp.replace('_', ' ')}", fontsize="large", fontweight="bold")
    fig.tight_layout()
    out_path_plot = os.path.join(dir_path, f"RMSF_{smp}.{fmt}")
    fig.savefig(out_path_plot)
    return out_path_plot


def rms(rms_type, traj, out_dir, sample_name, format_output, use_dots_for_rmsf, reference_frame, info=None,
        frames_lim=None, mask=None, atom_from_res=None, domains=None):
    """
    Compute the Root Mean Square Deviation or the Root Mean Square Fluctuation and create the plot.

    :param rms_type: the type of analysis, RMSD or RMSF.
    :type rms_type: str
    :param traj: the trajectory.
    :type traj: pt.Trajectory
    :param out_dir: the output directory path
    :type out_dir: str
    :param sample_name: the sample name.
    :type sample_name: str
    :param format_output: the output format for the plots.
    :type format_output: str
    :param use_dots_for_rmsf: if dots should be used to represent the RMSF value of each residue.
    :type use_dots_for_rmsf: bool
    :param reference_frame: the reference frame index (0 indexing).
    :type reference_frame: int
    :param info: the molecular dynamics information as free text.
    :type info: str
    :param frames_lim: the frames' limits.
    :type frames_lim: list
    :param mask: the applied mask.
    :type mask: str
    :param atom_from_res: the atom number corresponding to a residue number.
    :type atom_from_res: dict
    :param domains: the domains' coordinates and info.
    :type: Pandas.Dataframe
    :raises ValueError: unknown RMS type
    """
    log_txt = f"{rms_type} computation"
    if mask:
        log_txt = f"{log_txt}, with Mask \"{mask}\""
    if frames_lim:
        range_frames = [x for x in range(frames_lim[0], frames_lim[1])]
        log_txt = f"{log_txt}, selected frames {frames_lim[0]} to {frames_lim[1]}"
    else:
        range_frames = [x for x in range(traj.n_frames)]
    logging.info(f"{log_txt}, with frame {reference_frame} as reference:")

    path_csv = f"{os.path.join(out_dir, f'{rms_type}_{sample_name}')}.csv"

    subtitle_plot = None
    if mask:
        subtitle_plot = f"Applied mask: {mask}"
    if frames_lim:
        subtitle_plot = f"{subtitle_plot}{'    ' if subtitle_plot else ''}Frames used: {frames_lim[0]}-{frames_lim[1]}"
    subtitle_plot = f"{subtitle_plot}{'    ' if subtitle_plot else ''}Reference frame: {reference_frame}"
    if info:
        subtitle_plot = f"{subtitle_plot}{'    ' if subtitle_plot else ''}{info}"

    if rms_type == "RMSD":
        rmsd_traj = pt.rmsd(traj, mask=mask, ref=reference_frame, frame_indices=range_frames)
        source = pd.DataFrame({"frames": range_frames, f"{rms_type}": rmsd_traj})
        plot_line_path = plot_rmsd_line(source, sample_name, out_dir, format_output, subtitle_plot)
        plot_histogram_path = plot_rmsd_histogram(source, sample_name, out_dir, format_output, subtitle_plot)
        plot_path = f"{plot_line_path}, {plot_histogram_path}"
    elif rms_type == "RMSF":
        # superpose on the reference frame used for the trajectory
        traj_superpose = traj.superpose(ref=reference_frame, mask=mask)
        rmsf_traj = pt.rmsf(traj_superpose, mask=mask)
        tmp_source = pd.DataFrame({"atoms": rmsf_traj.T[0], f"{rms_type}": rmsf_traj.T[1]})
        source = rmsf_residues(tmp_source, atom_from_res)
        subtitle_plot = f"{subtitle_plot}\nAverage RMSF of the atoms by residues"
        plot_path = plot_rmsf(source, sample_name, out_dir, format_output, use_dots_for_rmsf, subtitle_plot,
                              src_domains=domains)
    else:
        raise ValueError(f"{rms_type} is not a valid case, only \"RMSD\" or \"RMSF\" are allowed.")
    source.to_csv(path_csv, index=False)
    logging.info(f"\tdata saved: {path_csv}")
    logging.info(f"\t{rms_type} plot{'s' if rms_type == 'RMSD' else ''} saved: {plot_path}")


if __name__ == "__main__":
    descr = f"""
    {os.path.basename(__file__)} v. {__version__}

    Created by {__author__}.
    Contact: {__email__}
    {__copyright__}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied.

    From a molecular dynamics trajectory file perform trajectory analysis. The script computes the Root Mean Square 
    Deviation (RMSD) and the Root Mean Square Fluctuation (RMSF).
    
    The reference frame for the RMSD and the RMSF is computed by clustering if no reference frame is defined with the 
    option '--ref-frame', in that case the PDB file of the most representative cluster's frame is saved in the results 
    directory. 
    """
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", required=True, type=str, help="the path to the output directory.")
    parser.add_argument("-s", "--sample", required=True, type=str, help="the sample ID used for the files names.")
    parser.add_argument("-t", "--topology", required=True, type=str,
                        help="the path to the molecular dynamics topology file.")
    parser.add_argument("-i", "--info", required=False, type=str,
                        help="the molecular dynamics simulation complementary information, as the MD simulation time. "
                             "Set as free text which will be added to the subtitle of the plots.")
    parser.add_argument("-r", "--ref-frame", required=False, type=int,
                        help="the reference frame index to use for the RMSD and the RMSF. The index of the reference "
                             "frame must be defined on the whole trajectory, not on the trajectory specified with "
                             "'--frames' option.")
    parser.add_argument("-f", "--frames", required=False, type=str,
                        help="the frames to use for the RMSD and the RMSF. The format must be two integers separated "
                             "by a colon (frames 500 to 2000, use --frames 500:2000), or from a frame to the end "
                             "(frames 500 to the end, --frames 500:) or from the first frame to a specified frame "
                             "(first frame to the frame 2000, --frames :2000).")
    parser.add_argument("-x", "--step", required=False, type=int, default=1,
                        help="the step to apply for the frames selection in the trajectory.")
    parser.add_argument("-d", "--domains", required=False, type=str, default="",
                        help="the path to the CSV domains file. The domains file is used in the RMSF plot to display a "
                             "map of the domains. If the mask do not cover all the domains in the file, the domains "
                             "argument should not be used. the domains file is a comma separated file, the first "
                             "column is the annotation name, the 2nd is the residue start coordinate, the 3rd is the "
                             "residue end coordinate, the last one is the color to apply in hexadecimal format. The "
                             "coordinate are 1-indexed.")
    parser.add_argument("-m", "--mask", required=False, type=str, default="",
                        help="the residues mask selection. To select a region in the domains CSV file, you can use a "
                             "mask specifying the domain's residues positions, i.e: ':60-238@CA,C,O,N'.")
    parser.add_argument("--dots-for-residues", required=False, action="store_true",
                        help="use dots on the RMSF plot for each residue, useful when a mask is used and not all the "
                             "protein residues are used.")
    parser.add_argument("-y", "--format", required=False, default="svg",
                        choices=["eps", "jpg", "jpeg", "pdf", "pgf", "png", "ps", "raw", "svg", "svgz", "tif", "tiff"],
                        help="the output plots format: 'eps': 'Encapsulated Postscript', "
                             "'jpg': 'Joint Photographic Experts Group', 'jpeg': 'Joint Photographic Experts Group', "
                             "'pdf': 'Portable Document Format', 'pgf': 'PGF code for LaTeX', "
                             "'png': 'Portable Network Graphics', 'ps': 'Postscript', 'raw': 'Raw RGBA bitmap', "
                             "'rgba': 'Raw RGBA bitmap', 'svg': 'Scalable Vector Graphics', "
                             "'svgz': 'Scalable Vector Graphics', 'tif': 'Tagged Image File Format', "
                             "'tiff': 'Tagged Image File Format'. Default is 'svg'.")
    parser.add_argument("--keep-pdb-first-frame", required=False, action="store_true",
                        help="if the first frame of the trajectory should be kept.")
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
    trajectory = None
    try:
        trajectory = load_trajectories(args.inputs, args.topology, args.info)
    except RuntimeError as exc:
        logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({args.inputs}) files exists",
                      exc_info=True)
        sys.exit(1)
    except ValueError as exc:
        logging.error(f"Check if the topology ({args.topology}) and/or the trajectory ({args.inputs}) files exists.",
                      exc_info=True)
        sys.exit(1)
    except IndexError as exc:
        logging.error(exc, exc_info=True)
        sys.exit(1)

    # check the trajectory limits
    frames_limits = None
    try:
        frames_limits = check_limits(args.frames, args.ref_frame, trajectory)
    except argparse.ArgumentTypeError as exc:
        logging.error(exc)
        sys.exit(1)

    # extracting .pdb file from the first frame of the trajectory
    pdb_first_frame_path = os.path.join(args.out, f"{args.sample.replace(' ', '_')}_frame-1.pdb")
    pt.write_traj(pdb_first_frame_path, traj=trajectory, overwrite=True, frame_indices=[0, 1])
    # get the atoms belonging to each residue from the .pdb file
    atom_res = link_atoms_to_residue_from_pdb(args.sample.replace(" ", "_"), pdb_first_frame_path)

    # get the most representative cluster
    if args.ref_frame is not None:
        logging.info(f"No clustering performed, frame {args.ref_frame} is used as reference.")
        ref_frame_idx = args.ref_frame
        if args.step:
            logging.warning(f"--step option used for clustering is ignored because --ref-frame {args.ref_frame} is "
                            f"defined.")
    else:
        ref_frame_idx = get_reference_frame(trajectory, args.mask, frames_limits, args.step,
                                            os.path.join(args.out, f"{args.sample.replace(' ', '_')}_cluster.pdb"))

    try:
        # compute RMSD and create the plot
        rms("RMSD", trajectory, args.out, args.sample.replace(" ", "_"), args.format, args.dots_for_residues,
            ref_frame_idx, args.info, frames_limits, args.mask)
        # compute RMSF and create the plot
        rms("RMSF", trajectory, args.out, args.sample.replace(" ", "_"), args.format, args.dots_for_residues,
            ref_frame_idx, args.info, frames_limits, args.mask, atom_res, domains_data)
    except ValueError as exc:
        logging.error(exc, exc_info=True)
        sys.exit(1)

    if not args.keep_pdb_first_frame:
        os.remove(pdb_first_frame_path)
        logging.info("Extracted PDB file from the first frame of the trajectory removed.")
