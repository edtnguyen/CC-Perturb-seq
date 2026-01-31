#!/usr/bin/env python
# 2021-06-16 RTK; Simplify from fastq_bc_sep.py
# 2021-08-10 RTK; v0.2 set gzip compress level to 4 (only mod; hardcoded)
# 2022-06-21 RTK; v0.4 Update for kits WT, WT_mini, WT_mega
# 2024-04-22 RTK; v0.5 Update for chem v1-v3, new barcodes (pipeline v1.3.0)
# 2026-01-31; v0.6 Load barcodes from ../barcodes; v1-v2 kits only
#
# Separate fastq reads by well group
#
# Barcode data is loaded from ../barcodes folder (relative to this script)
#

import argparse
import os
import sys
import gzip
import json
import datetime
import re
import numpy as np
import pandas as pd


__version__ = "V0.6; 2026-01-31"
VERSION = os.path.basename(__file__) + " " + __version__


DEF_BC_EDIT_DIST = 2

BC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "barcodes"))

GZIP_COMPRESSLEVEL = 4
GZIP_REC_BUFFER = 1000

UPDATE_FREQ = 100000

# Sample size just used for kit scoring (no barcode correction)
BC_SAMP_SIZE = 500000
BC_SAMP_SKIP = 100000
KIT_SCORE_MIN = 0.4

DEF_OPREF = "group"

DNA_BASES = set(list("ACGT"))
DNA_REGEX = re.compile(r"[ACGT]+$")


# Story
def explain_details():
    story = f"""
-------------------------------------------------------------------------------
{VERSION}
-------------------------------------------------------------------------------
Description

    Separate reads in fastq files by sample well groups.

    Typical use:

        fastq_sep_groups.py -f <R1.fastq.gz> -c <chemistry> -g [sample well groups]...

    -------------------
    Input fastq files should be gzip compressed. If only given R1 filename
    (via -f or --fq1), R2 is filename is deduced from that.

    -------------------
    The chemistry version for the data must be given (via -c or --chemistry).

    Kit is normally guessed via score. However, this may be given explicitly.
    If a given kit does not match the guessed kit, or if the best kit score
    is too low, kit score checking may be bypassed (via --kit_score_skip)
    To simply check kit (and not process fastqs), run with --dryrun option.

    -------------------
    Barcode data is loaded from ../barcodes (relative to this script). Use
    --bcpath to override. The folder should contain matching bc_data_*.csv and
    bc_dict_*.json files.

    -------------------
    Optional barcode replacement file (--bc_replace) format:
        <left_bc> *<right_bc>
    Reads with bc1 matching right_bc (after edit-distance correction) are mapped
    to left_bc before well assignment. Invalid entries are warned and skipped.

    -------------------
    Sample well groups are specified by <name> and <wells>.

    Wells are specified in blocks, ranges, or individually like this:
        'A1:C6' specifies a block as [top-left]:[bottom-right]; A1-A6, B1-B6, C1-C6.
        'A1-B6' specifies a range as [start]-[end]; A1-A12, B1-6.
        'C4' specifies a single well.
        Multiple selections are joined by commas (no space), e.g. 'A1-A6,B1:D3,C4'

    Groups may be specified via command line or from a list file (--gfile <list>).

    For group specification via command line, each group requires one command:
        --group <name1> <wells-for-group1>
        --group <name2> <wells-for-group2>
        ... etc ...

    For group specification via file (--gfile <list>), a simple, space delimited
    text list file is used. Each file line should have:
        '<name1> <wells1>'
        '<name2> <wells2>'
        ... etc ...

"""
    print(story)


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Splits fastq reads by wells")
    parser.add_argument("-f", "--fq1", help="R1 fastq.gz file (must be gzip'd)")
    parser.add_argument("--fq2", help="R2 fastq.gz file (must be gzip'd)")
    parser.add_argument("-c", "--chemistry", help="Set chemistry version for data")
    parser.add_argument(
        "-g", "--gfile", help="Group specifications from file; See --explain for format"
    )
    parser.add_argument(
        "--group",
        action="append",
        nargs=2,
        metavar=("NAME", "WELLS"),
        help="Add group name and well specification; See '--explain' for format",
    )
    parser.add_argument(
        "--bc_replace",
        help="Barcode replacement file; format: <left_bc> *<right_bc>",
    )
    parser.add_argument("--obase", help="Output file basename")
    parser.add_argument("--opath", help="Output file path; Will create if needed")
    parser.add_argument(
        "--bcpath",
        help="Barcode folder path; default is ../barcodes relative to this script",
    )
    parser.add_argument(
        "--only_stats",
        default=0,
        action="store_true",
        help="Only report stats; No fastq outputs",
    )
    parser.add_argument(
        "--range",
        type=int,
        nargs="+",
        help="Limit range of reads as [Last] or [First, Last]",
    )
    parser.add_argument(
        "--kit", help="Kit used; This is normally guessed from the data"
    )
    parser.add_argument(
        "--kit_score_skip",
        action="store_true",
        help="Ignore kit score failure; WARNING Use with caution!",
    )
    parser.add_argument(
        "--dryrun", action="store_true", help="Dry run; Only setup and report status"
    )
    parser.add_argument(
        "--chem_list",
        action="store_true",
        help="List valid chemistry versions and kit names",
    )
    parser.add_argument(
        "-e",
        "--explain",
        default=0,
        action="store_true",
        help="Explain some assumptions and details",
    )
    parser.add_argument(
        "-V", "--version", action="version", version="%(prog)s " + __version__
    )
    args = parser.parse_args()

    if args.explain:
        explain_details()
        return True

    if args.chem_list:
        report_valid_kits()
        return True

    if not check_options(args):
        print("Problem with options")
        return False

    # Set amp seq
    args.amp_seq = get_samp_seq(args.chemistry)
    print("# Set amplicon sequence for chemistry {args.chemistry}")

    time_st = datetime.datetime.now()
    print(f"# Start time {time_st}")
    print_now("# Initializing...")

    # Collection of all barcode data
    all_bc_dict = bc_dir_to_bc_dict(args.bcpath)
    if not all_bc_dict:
        print("Problem loading barcode data")
        return False
    print_now(f"# Loaded barcode data ({len(all_bc_dict)} bc sets)")
    # Sample of bc fastq
    fq2_df = fastq_bc_seqs_df(args.fq2, args.amp_seq)
    print_now("# Loaded fastq R2 sample")

    # Get guessed kit and score for this
    s_kit, k_score = guess_kit_from_bc(args, all_bc_dict, fq2_df)
    if k_score < args.kit_score_min:
        print(f"Problem guessing kit; Score {k_score} < min {args.kit_score_min}")
        if args.kit_score_skip:
            print("\nWARNING; kit_score_skip true, so continuing...\n")
        else:
            return False
    # Check if given matches guessed
    if args.kit:
        if args.kit != s_kit:
            print(f"Problem kit mismatch given '{args.kit}' vs guessed '{s_kit}'")
            if args.kit_score_skip:
                print("\nWARNING; kit_score_skip true, so continuing...\n")
            else:
                return False
    if not args.kit:
        args.kit = s_kit

    bc_info = get_barcode_info(args, all_bc_dict)
    if not bc_info:
        print("Problem getting barcodes")
        return False
    # Optional barcode replacements
    bc_replace_map = load_bc_replace_map(
        args.bc_replace, set(bc_info["bc_seqs"][1])
    )
    if bc_replace_map is None:
        print("Problem loading barcode replacement file")
        return False

    # Get input fastq file handles
    fq1_file, fq2_file = get_fastq_files(args)
    if not fq1_file:
        print("Problem with fastq files")
        return False

    # Get output sample data
    out_sets = get_outsets(args, bc_info)
    if not out_sets:
        print("Problem with group / well subset specification")
        return False

    # Set up outputs (path, basename), update output set objs with this
    if not set_up_full_fnames(args, out_sets):
        print("Problem setting up filenames")
        return False

    # Status; maybe only
    report_settings(args, bc_info, out_sets)
    if args.dryrun:
        print("Dry run only; All done")
        return True

    # Get mapping of well/cell indices to output objects
    bci_oset_dict = {}
    for oset in out_sets:
        for index in oset.get_iset():
            if index in bci_oset_dict:
                bci_oset_dict[index].append(oset)
            else:
                bci_oset_dict[index] = [oset]

    # Get various working vars
    s2_min_len = len(bc_info["amp_seq"])
    bc_max_edist = args.edit_dist
    bc_dicts = bc_info["bc_dicts"]
    bc_starts = bc_info["bc_starts"]
    bc_ends = bc_info["bc_ends"]
    # Mapping of bc seqs to well index (1-based int)
    bc1_bc_seq_to_wind = bc_info["bc_seq_to_wind"][1]

    # Init counter
    read_stats = {
        "number_of_reads": 0,
        "reads_too_short": 0,
        "reads_valid_bc": 0,
        "reads_ambig_bc1": 0,
        "bc1_Q30": 0,
        "bc2_Q30": 0,
        "bc3_Q30": 0,
        "bc_edit_dist_NA": 0,
        "bc_edit_dist_0": 0,
        "bc_edit_dist_1": 0,
        "bc_edit_dist_2": 0,
        "index_with_outs": 0,
        "index_mult_outs": 0,
        "index_no_outs": 0,
        "total_outputs": 0,
    }

    # Run through fastq records
    print_now("# Processing reads...")
    n_rec = 0
    while True:
        head1 = fq1_file.readline()
        seql1 = fq1_file.readline()
        plus1 = fq1_file.readline()
        qual1 = fq1_file.readline()
        head2 = fq2_file.readline()
        seql2 = fq2_file.readline()
        plus2 = fq2_file.readline()
        qual2 = fq2_file.readline()
        # All done?
        if (not seql1) or (not seql2):
            break
        n_rec += 1
        # Before or after range?
        if n_rec < args.rec_first:
            continue
        if args.rec_last and (n_rec > args.rec_last):
            break
        read_stats["number_of_reads"] += 1

        if (n_rec % UPDATE_FREQ) == 0:
            print_now(f"# record {n_rec}")

        header1 = head1.decode().strip()
        header2 = head2.decode().strip()
        # Names must match
        if header1.split()[0] != header2.split()[0]:
            story = f"Fastq R1 R2 name mismatch at record {n_rec}\n"
            story += f"Fastq files: {fq1_file.name} {fq2_file.name}\n"
            story += f"Header fq1: {header1}\n"
            story += f"Header fq2: {header2}"
            print(story)

        # Get barcodes (actually only bc1)
        seq2 = seql2.decode().strip()
        if len(seq2) < s2_min_len:
            read_stats["reads_too_short"] += 1
            continue

        rbc1 = seq2[bc_starts[1] : bc_ends[1]]
        if "N" in rbc1:
            read_stats["reads_ambig_bc1"] += 1

        # Find matching (perfect) barcodes for raw bc1
        mat_list, ed_max, found = get_min_edit_dists(rbc1, bc_dicts[1], bc_max_edist)
        if not found:
            ed_max = "NA"
        ed_key = f"bc_edit_dist_{ed_max}"
        read_stats[ed_key] += 1
        # valid if any found
        if mat_list:
            read_stats["reads_valid_bc"] += 1
        else:
            continue
        # Apply barcode replacements after edit-distance correction
        if bc_replace_map:
            mat_list = [bc_replace_map.get(bc, bc) for bc in mat_list]

        # Well index for barcodes; Use set in case multiple possibilities
        match_bci_set = set()
        for bc in mat_list:
            wind = bc1_bc_seq_to_wind[bc]
            match_bci_set.add(wind)

        # Handle all sample matches; Get unique set of outputs (groups)
        u_osets = set()
        for bc_idx in match_bci_set:
            if bc_idx in bci_oset_dict:
                for oset in bci_oset_dict[bc_idx]:
                    u_osets.add(oset)

        # Process unique outputs
        if u_osets:
            read_stats["index_with_outs"] += 1
            if len(u_osets) > 1:
                read_stats["index_mult_outs"] += 1
            for oset in u_osets:
                read_stats["total_outputs"] += 1
                oset.inc_count()
                if args.only_stats:
                    continue
                # Object handles writing
                if not oset.write_rec(
                    head1, seql1, plus1, qual1, head2, seql2, plus2, qual2
                ):
                    print(f"Problem writing output for {oset.get_name()}; Bailing")
                    print(f"Number reads so far {read_stats['number_of_reads']}")
                    print(oset)
        else:
            read_stats["index_no_outs"] += 1
    # Stats
    read_stats = add_Q30_stats(read_stats, fq2_df)
    print("=" * 70)
    for k, v in read_stats.items():
        print(f"{k}\t{v}")
    print()
    for oset in out_sets:
        if args.only_stats:
            print(f"Sample {oset.get_name()} total\t{oset.get_count()}")
        else:
            oset.close_ofiles(verb=True)
    print()

    time_en = datetime.datetime.now()
    print("Total time", str(time_en - time_st)[:-4])
    print()

    return True


# ------------------------- Chem and Kit stuff -------------------------------
#
# Dummy func
def is_custom_kit(kit):
    return False


KIT_CHEM_DEFS = """
    kit         chem  rows    cols    plates  bc1           bc2     bc3    ktype
    WT_mini     v1    1       12      1       n24_v4        v1      v1     normal
    WT          v1    4       12      1       v2            v1      v1     normal
    WT_mega     v1    8       12      1       n198_v5       v1      v1     normal
    WT_mini     v2    1       12      1       n24_v4        v1      v1     normal
    WT          v2    4       12      1       n99_v5        v1      v1     normal
    WT_mega     v2    8       12      1       n198_v5       v1      v1     normal
"""


def _make_kit_chem_tab_df():
    """Get kit / chemistry table as dataframe"""
    head_row = []
    rows = []
    for line in KIT_CHEM_DEFS.split("\n"):
        parts = line.split("#")[0].split()
        if parts:
            if head_row:
                rows.append(parts)
            else:
                head_row = parts

    kit_df = pd.DataFrame(rows, columns=head_row)
    num_cols = "rows,cols,plates".split(",")
    kit_df[num_cols] = kit_df[num_cols].apply(pd.to_numeric, axis=1)
    kit_df["nwells"] = kit_df[num_cols].product(axis=1)
    return kit_df


# Create once as global
KIT_CHEM_TAB = _make_kit_chem_tab_df()


def get_kit_chem_ddata(kit, chem="v2"):
    """Get definition data for given kit and chemistry

    Return dict
    """
    new_dict = {}
    chem = parse_chemistry(chem)
    kit, _ = parse_kit(kit, chem=chem)
    query = f"kit == '{kit}' and chem == '{chem}'"
    drow = KIT_CHEM_TAB.query(query)
    if len(drow):
        new_dict = drow.iloc[0].to_dict()
    return new_dict


def parse_chemistry(chem, as_num=False):
    """Parse given chemistry to standard form

    Return str
    """
    answer = ""
    # Get number
    ok, n = str_to_num(chem, regex=True)
    if ok:
        if as_num:
            answer = n
        else:
            chem = f"v{n}"
            if chem in KIT_CHEM_TAB["chem"].values:
                answer = chem
    return answer


def parse_kit(kit, chem=None, unique=True):
    """Attempt to parse given kit into cannonical form

    Returns cannonical named kit and story

    If name without chem is ambiguous and unique flag, fail

    Return tuple (str, str)
    """
    # Make sure kit is string (e.g. in case given int)
    kit_s = str(kit)
    # If given chemistry, make sure it's good
    if chem:
        chem_s = parse_chemistry(chem)
        if not chem_s:
            return None, f"Bad chemistry ({chem}) given for parse_kit({kit})"
        chem = chem_s

    # If no chemistry, unique kit-chem not required
    if not chem:
        unique = False

    story = ""
    k_list = []
    df = KIT_CHEM_TAB

    # Try to parse number first; If not int, empty list
    k_list = kit_chem_from_int(kit_s, chem=chem)
    if len(k_list) < 1:
        # Lowercase and maybe dash not underscore (e.g. WT-mega >--> WT_mega)
        kit_s = kit_s.replace("-", "_").lower()
        df = df[df["kit"].str.lower() == kit_s]
        if chem:
            df = df[df["chem"] == str(chem)]
        k_list = [list(t) for t in df[["kit", "chem"]].values]

    # Different story for different number of matches
    if len(k_list) < 1:
        kit_s = ""
        story = f"Kit '{kit}' unrecognized"
    elif len(k_list) > 1:
        if unique:
            kit_s = ""
            story = f"Kit '{kit}' ambiguous; [kit,chem] = {k_list}"
        else:
            # First one, as all in list match except for chem
            kit_s, chem = k_list[0]
    else:
        kit_s, chem = k_list[0]

    return kit_s, story


def kit_chem_from_int(kint, chem=None, verb=True):
    """Get kit from int based on number of wells (e.g. 48 >--> WT), maybe chemistry

    chem = chemistry specification

    Return list of lists [[kit,chem], [kit,chem], ...]
    """
    if is_custom_kit(kint):
        chem = ""

    k_list = []
    # If given chemistry, make sure it's good
    if chem:
        chem_s = parse_chemistry(chem)
        if not chem_s:
            if verb:
                print(f"Bad chemistry ({chem}) given for kit_chem_from_int({kint})")
            return k_list
        chem = chem_s

    # Make sure starting with int; No regex, only simple int should work
    ok, n = str_to_num(kint, regex=False)
    if ok:
        # Subset of rows by wells (int) and maybe chem (str)
        df = KIT_CHEM_TAB[KIT_CHEM_TAB["nwells"] == n]
        if chem:
            df = df[df["chem"] == chem]
        k_list = [list(t) for t in df[["kit", "chem"]].values]

    return k_list


def kit_name_list(chem="v2", as_list=False, ktype="normal"):
    """Get list of kits

    as_list = flag to return list with [kit,chem] items
    ktype = filter for kit type column

    Return list[kit,] or list of [[kit, chem],]
    """
    # Filter table on any restrictions
    df = KIT_CHEM_TAB
    if chem:
        chem = parse_chemistry(chem)
        df = df[df["chem"] == str(chem)]
    if ktype:
        df = df[df["ktype"] == ktype]
    # Returning list of only kit or [kit, chem]
    if as_list:
        k_list = [list(t) for t in df[["kit", "chem"]].values]
    else:
        k_list = list(df["kit"].unique())
    return k_list


def kit_chem_list(kit):
    """Get list of chemistry for kit

    Return list[chem]
    """
    kit, _ = parse_kit(kit, chem=None)
    df = KIT_CHEM_TAB
    df = df[df["kit"] == kit]
    return list(df["chem"])


def kit_num_wells(kit, chem):
    """Get number of wells for kit

    Return int
    """
    if is_custom_kit(kit):
        return 1

    kit, _ = parse_kit(kit, chem=chem)
    # Filter table on any restrictions
    df = KIT_CHEM_TAB
    df = df[df["kit"] == kit]
    if chem:
        chem = parse_chemistry(chem)
        df = df[df["chem"] == str(chem)]
    # Only get number if unambiguous
    k_set = set(df["nwells"])
    nwell = list(k_set)[0] if len(k_set) == 1 else 0

    return nwell


def report_valid_kits(chem=None, ktype="normal", pad=True):
    """Report kit collection;

    If not given chemistry, report that too

    Returns nothing
    """
    # List of kit names, possibly limited to chem and ktype
    k_list = kit_name_list(as_list=False, chem=chem, ktype=ktype)

    if pad:
        print()
    print(f"There are {len(k_list)} installed kits:\n")
    for kit in k_list:
        n_wells = kit_num_wells(kit, chem)
        c_list = kit_chem_list(kit)
        chemistry = ", ".join(c_list)
        print(f"    {kit:12s} {n_wells} wells,   Chemistry: {chemistry}")
    if pad:
        print()


def kit_bc_set_list(kit, chem, as_tup=False):
    """Get list of barcode set names for kit and chemistry

    kit = kit name
    chem = chemistry version
    as_tup = flag to return list of [bcX, name] items vs names

    Return list of barcode names per round
    """
    bc_list = []
    # Parse first
    chem = parse_chemistry(chem)
    kit_s, _ = parse_kit(kit, chem=chem)
    if kit_s and chem:
        df = KIT_CHEM_TAB
        row = df[(df["kit"] == kit_s) & (df["chem"] == str(chem))]
        cols = [f"bc{r}" for r in "123"]
        bc_list = list(df.loc[row.index, cols].values.flatten())
        # Tuple as [bcX, name]? This form matches 'bc_round_set' param
        if as_tup:
            new_list = []
            for col, bc in zip(cols, bc_list):
                new_list.append([col, bc])
            bc_list = new_list
    return bc_list


def kit_round1_bc_dict(chem):
    """Get dict mapping kit to name of round1 barcode set name

    Return dict[kit] = r1bc
    """
    new_dict = {}
    k_list = kit_name_list(as_list=True, chem=chem)
    for kit, chem in k_list:
        bc_list = kit_bc_set_list(kit, chem)
        new_dict[kit] = bc_list[0]
    return new_dict


# ----------------------- options / filenames / feedback ---------------------
#
def set_def_options(comargs):
    # Hardcode options
    comargs.edit_dist = DEF_BC_EDIT_DIST
    comargs.no_opref = False
    comargs.kit_score_min = KIT_SCORE_MIN


def check_options(comargs):
    """Check options and make consistent

    Some values inserted into comargs structure
    """
    set_def_options(comargs)

    # Input filenames
    if not comargs.fq1:
        print("Problem: Need to specify fastq files; Try --help or --explain")
        return False
    # fq2 maybe from fq1
    if not comargs.fq2:
        comargs.fq2 = get_fq2_from_fq1(comargs.fq1)
    if not comargs.fq2:
        print("Problem: Need both R1 and R2 fastq files; Try --help or --explain")
        return False

    # Check files are good; Also reports if bad and expands env vars
    comargs.fq1 = check_infile(comargs.fq1, verb=True)
    comargs.fq2 = check_infile(comargs.fq2, verb=True)
    if (not comargs.fq1) or (not comargs.fq2):
        print("Problem with input files")
        return False

    # Edit distance
    if not 0 <= comargs.edit_dist <= 2:
        print(f"Problem: Bad edit distance {comargs.edit_dist}; Allowed 0 - 2")
        return False

    # Group spec given or from file
    if comargs.gfile:
        comargs.gfile = check_infile(comargs.gfile, verb=True)
        if not comargs.gfile:
            print("Problem with group specification list")
            return False
    if (not comargs.gfile) and (not comargs.group):
        print("Problem: Need to specify groups")
        return False

    # Given chem and any kit must be legit
    if not comargs.chemistry:
        print("Problem: Need to specify chemistry")
        return False
    chem = parse_chemistry(comargs.chemistry)
    if not chem:
        print("Problem with given chemistry '{comargs.chemistry}'")
        return False
    comargs.chemistry = chem

    if comargs.kit:
        kit_n, kit_s = parse_kit(comargs.kit)
        if not kit_n:
            print("Problem with given kit '{comargs.kit}'")
            report_valid_kits()
            return False

    # Barcode folder
    if comargs.bcpath:
        comargs.bcpath = os.path.expanduser(os.path.expandvars(comargs.bcpath))
    if comargs.bc_replace:
        comargs.bc_replace = check_infile(comargs.bc_replace, verb=True)
        if not comargs.bc_replace:
            print("Problem with barcode replacement file")
            return False

    # Ouptut filename prefix
    if comargs.no_opref:
        comargs.opref = ""
    else:
        comargs.opref = DEF_OPREF

    # Range of records to process
    ok, first, last = parse_range_list(comargs.range)
    if not ok:
        print(f"Bad range given; {comargs.range}")
        return False
    comargs.rec_first = first
    comargs.rec_last = last

    return True


def get_samp_seq(chem):
    """Get chem-specific default parameters

    Return seq
    """
    # Read structure
    # v3 chemistry
    if parse_chemistry(chem, as_num=True) == 3:
        seq = "NNNNNNNNNN33333333ATGAGGGGTCAG22222222TCCAACCACCTC11111111"
    else:
        seq = "NNNNNNNNNN33333333GTGGCCGATGTTTCGCATCGGCGTACGACT22222222ATCCACGTGCTTGAGACTGTGG11111111"
    return seq


def check_infile(fname, verb=True, toxic=False):
    """Check file

    Return filename, possibly expanded if env var
    """
    r_fname = ""
    # Expand any env vars; User for ~/ and vars for $home
    fname = os.path.expanduser(fname)
    fname = os.path.expandvars(fname)
    # Read access means input is good
    if os.access(fname, os.R_OK):
        r_fname = fname
    else:
        story = f"Cannot read file: '{fname}'"
        if toxic:
            raise IOError(story)
        if verb:
            print(story)
    return r_fname


def parse_range_list(rlis, warn=True):
    """Parse range list; maybe [], [end], [start,end]

    first and last default zero; If given, set as int

    Return status,first,last
    """
    ok = True
    first = last = 0
    if rlis:
        if len(rlis) > 1:
            if (len(rlis) > 2) and warn:
                print(f"Extra range args ignored; {rlis}")
            ok, first = str_to_num(rlis[0])
            if ok:
                ok, last = str_to_num(rlis[1])
        else:
            ok, last = str_to_num(rlis[0])
    return ok, first, last


def range_story(r_ends):
    first = last = ""
    if r_ends:
        if len(r_ends) > 1:
            first, last = r_ends[:2]
        else:
            last = r_ends[0]
    if first and last:
        story = f"From {first} to {last}"
    elif last:
        story = f"To max {last}"
    else:
        story = "No restrictions"
    return story


# --------------------------- Load barcode data ------------------------------
#
def load_bc_replace_map(fname, bc1_set=None):
    """Load bc1 replacement map

    Format: <left_bc> *<right_bc>
    Returns dict mapping right_bc -> left_bc
    """
    if not fname:
        return {}

    bc1_set = set([b.upper() for b in bc1_set]) if bc1_set else None
    repl_map = {}

    with open(fname) as infile:
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"Warning: bad bc_replace line {line_num}; skipping")
                continue
            left = parts[0].strip().upper()
            right = parts[1].strip().upper()
            if right.startswith("*"):
                right = right[1:]
            if not (DNA_REGEX.match(left) and DNA_REGEX.match(right)):
                print(f"Warning: bad bc_replace line {line_num}; skipping")
                continue
            if bc1_set and left not in bc1_set:
                print(
                    f"Warning: bc_replace target '{left}' not in bc1 set; skipping"
                )
                continue
            if bc1_set and right not in bc1_set:
                print(
                    f"Warning: bc_replace source '{right}' not in bc1 set; skipping"
                )
                continue
            if right in repl_map and repl_map[right] != left:
                print(
                    f"Warning: bc_replace duplicate for '{right}'; using latest mapping"
                )
            repl_map[right] = left

    return repl_map


def bc_dir_to_bc_dict(bc_dir=BC_DIR, bc_sets=None):
    """Load all barcode data structs from bc_data_*.csv and bc_dict_*.json

    Return dict with [name] = {'data': data, 'dict': dict}
    """
    if not bc_dir:
        bc_dir = BC_DIR

    bc_dir = os.path.expanduser(os.path.expandvars(bc_dir))
    bc_dir = os.path.abspath(bc_dir)
    if not os.path.isdir(bc_dir):
        print(f"Problem: barcode folder not found: {bc_dir}")
        return None

    if bc_sets is None:
        bc_sets = set(KIT_CHEM_TAB[["bc1", "bc2", "bc3"]].values.flatten())
    bc_sets = {s for s in bc_sets if s}

    if bc_sets:
        missing_pairs = []
        for name in sorted(bc_sets):
            data_path = os.path.join(bc_dir, f"bc_data_{name}.csv")
            dict_path = os.path.join(bc_dir, f"bc_dict_{name}.json")
            missing = []
            if not os.path.exists(data_path):
                missing.append("data")
            if not os.path.exists(dict_path):
                missing.append("dict")
            if missing:
                missing_pairs.append(f"{name} ({', '.join(missing)})")
        if missing_pairs:
            print(
                "Problem: missing barcode files for required sets: "
                + ", ".join(missing_pairs)
            )
            return None

    data_files = [
        f for f in os.listdir(bc_dir) if f.startswith("bc_data_") and f.endswith(".csv")
    ]
    if not data_files:
        print(f"Problem: no bc_data_*.csv files found in {bc_dir}")
        return None

    new_dict = {}
    for fname in sorted(data_files):
        name = fname[len("bc_data_") : -len(".csv")]
        if bc_sets and name not in bc_sets:
            continue
        data_path = os.path.join(bc_dir, fname)
        dict_path = os.path.join(bc_dir, f"bc_dict_{name}.json")

        if not os.path.exists(dict_path):
            print(f"Warning: missing barcode dict for {name}; skipping")
            continue

        try:
            bc_df = pd.read_csv(data_path)
        except Exception as e:
            print(f"Problem reading barcode data file: {data_path}")
            print(e)
            return None

        if "bci" in bc_df.columns:
            try:
                bc_df["bci"] = bc_df["bci"].astype(int)
                bc_df = bc_df.set_index("bci", drop=True)
            except Exception:
                print(
                    f"Warning: could not set 'bci' index for {data_path}; using default index"
                )
        else:
            print(f"Warning: 'bci' column missing in {data_path}; using default index")

        try:
            with open(dict_path) as dfile:
                bc_dict = json.load(dfile)
        except Exception as e:
            print(f"Problem reading barcode dict file: {dict_path}")
            print(e)
            return None

        new_dict[name] = {"data": bc_df, "dict": bc_dict}

    if bc_sets:
        missing = sorted([bc for bc in bc_sets if bc not in new_dict])
        if missing:
            print(f"Problem: missing barcode sets: {', '.join(missing)}")
            return None

    return new_dict


def fastq_bc_seqs_df(
    fastq, amp_seq, samp_size=BC_SAMP_SIZE, samp_skip=BC_SAMP_SKIP, verb=True
):
    """Get dataframe sample of per-sequence barcodes

    Return dataframe with barcode seq and qual
    """
    if verb:
        print(f"# Sampling {samp_size} reads from {fastq}")

    min_seq_len = len(amp_seq)
    bc_starts, bc_ends = get_amp_part_bounds(amp_seq)

    seqs = []
    quals = []
    n_rec = 0
    n_read = 0
    with gzip.open(fastq) as f2_file:
        while n_read < samp_size:
            f2_file.readline()
            seq = f2_file.readline().decode()[:-1]
            f2_file.readline()
            qual = f2_file.readline().decode()[:-1]

            if not seq:
                break
            n_rec += 1

            if n_rec <= samp_skip:
                continue

            if len(seq) >= min_seq_len:
                n_read += 1
                seqs.append(seq)
                quals.append(qual)

    if verb:
        print(f"# Collected {n_read} data from {n_rec} fastq records")

    seqs = pd.Series(seqs)
    quals = pd.Series(quals)

    bc_df = pd.DataFrame()
    # Extract subseqs
    # bc_df['seq'] = seqs
    bc_df["bc1"] = seqs.str.slice(bc_starts[1], bc_ends[1])
    bc_df["bc2"] = seqs.str.slice(bc_starts[2], bc_ends[2])
    bc_df["bc3"] = seqs.str.slice(bc_starts[3], bc_ends[3])

    # bc_df['qual'] = quals
    bc_df["qc1"] = quals.str.slice(bc_starts[1], bc_ends[1])
    bc_df["qc2"] = quals.str.slice(bc_starts[2], bc_ends[2])
    bc_df["qc3"] = quals.str.slice(bc_starts[3], bc_ends[3])

    return bc_df


# ---------------------------- Kit guess from data ---------------------------
#
def guess_kit_from_bc(comargs, all_bc_dict, fq2_df):
    """Guess kit via score against barcode sets

    Return tuple (kit, best_score)
    """
    chem = comargs.chemistry
    kit_bc_names = kit_round1_bc_dict(chem)
    print(f"# Scoring fastq data against {len(kit_bc_names)} kits, chemistry {chem}")

    # R2 sample barcode seq counts (doesn't have to be log10, but tuned for that)
    counts = np.log10(fq2_df["bc1"].value_counts())

    # Screen kits and barcode sets
    best_kit = best_score = None
    for kit, bc in kit_bc_names.items():
        # Set of sequences
        bc_df = all_bc_dict[bc]["data"]
        bc_set = set(bc_df["sequence"])
        score = bc_count_kit_score(counts, bc_set)
        if (not best_score) or (score > best_score):
            best_score = score
            best_kit = kit
        print(f"#   {kit} ({bc}) = {round(score,3)}")
    print(f"# Best scoring kit = {best_kit}, {round(best_score, 3)}")

    return best_kit, best_score


def bc_count_kit_score(counts, bc_set, remain_frac=0.2, verb=False):
    """Calculate the score of fastq barcode seq counts for given barcode set

    Return score
    """
    # Split counts into first and remaining parts
    # First part of counts = top N-bc counts
    f_tot = len(bc_set)
    f_counts = counts.iloc[:f_tot]
    f_num = len(set(f_counts.index) & bc_set)
    f_frac = f_num / f_tot
    f_mean = f_counts.mean()

    # Remaining part of counts via slice and threshold
    r_counts = counts.iloc[f_tot:]
    # Thresh = fraction of values in first part
    r_thresh = f_mean * remain_frac
    r_counts = r_counts[r_counts >= r_thresh]
    r_tot = len(r_counts.index)
    # None case
    if r_tot < 1:
        r_num = 0
        r_mean = r_frac = 0
    else:
        r_num = len(set(r_counts.index) & bc_set)
        r_frac = r_num / r_tot
        r_mean = r_counts.mean()

    if verb:
        print(f_tot, f_mean, r_thresh, sep="\t")
        print(f_num, f_frac, f_mean, sep="\t")
        print(r_num, r_frac, r_mean, sep="\t")

    score = f_frac - r_frac
    return score


# ------------------------------- misc utils ---------------------------------
#
def get_fq2_from_fq1(fq1):
    """Try to get fq2 (R2) name from fq1 (R1) filename"""
    print("# Using R1 to get R2 fastq")
    fq2 = ""
    parts = fq1.split("R1")
    # Make sure 'R1' is in name and only sub last instance
    #   e.g. '/pathR1/examR1.fq' >--> '/pathR1/examR2.fq'
    if len(parts) > 1:
        fq2 = "R1".join(parts[:-1]) + "R2" + parts[-1]
    return fq2


# Matches float/int with optional sign and chars before / after
#   This part (?<![+-]) is not a capture; Instead prevents \D matching +-
#   From https://stackoverflow.com/questions/58841472/regex-for-digits-and-plus-minus-sign
NUM_REGEX = re.compile(r"\D*(?<![+-])([+-]?([0-9]*)(\.([0-9]+))?)\D*$")


def str_to_num(string, as_float=False, commas=True, regex=False):
    """Attempt to get number from string

    Return tuple (success, value)
    """
    # Optional clean up / extraction
    if isinstance(string, str):
        # Strip any commas?
        if commas:
            string = string.replace(",", "")
        # regex; Allows surrounding chars (e.g. 'abc-122.4x' >--> -122.4)
        if regex:
            mat = NUM_REGEX.match(string)
            if mat:
                string = mat[1]
    # Cast
    try:
        # Cast to float first regardless; then maybe int
        n = float(string)
        if not as_float:
            n = int(string)
        ok = True
    except Exception:
        ok = False
        n = string
    return ok, n


def set_up_full_fnames(comargs, out_sets):
    """Set up output filename stuff"""
    # Basename
    if not comargs.obase:
        # Basename
        comargs.obase = comargs.fq1.split("/")[-1].split(".")[0]
        # Also trim if it's got R1
        comargs.obase = comargs.obase.split("R1")[0]
        if comargs.obase[-1] == "_":
            comargs.obase = comargs.obase[:-1]

    # Prefix extends basename
    if comargs.opref:
        comargs.obase = f"{comargs.obase}_{comargs.opref}"
        if comargs.obase[-1] == "_":
            comargs.obase = comargs.obase[:-1]

    # Output path
    if comargs.opath:
        # Only try to create if doesn't exist
        if not os.path.exists(comargs.opath):
            try:
                os.makedirs(comargs.opath)
            except Exception as e:
                print(f"Problem creating subdir {comargs.opath}", e)
                return False
        if comargs.opath[-1] != "/":
            comargs.opath = comargs.opath + "/"
    else:
        comargs.opath = ""

    # Update outset object filenames
    for oset in out_sets:
        oset.set_fnames(obase=comargs.obase, opath=comargs.opath)

    return True


def print_now(story, newline=True, file=sys.stdout):
    """Print plus fulsh buffer"""
    if newline:
        print(story, file=file)
    else:
        print(story, file=file, end="")
    file.flush()


# ------------------------ Set up fastq / barcodes ---------------------------
#
def get_fastq_files(comargs):
    """Get and open fastq files

    Return tuple of file handles
    """
    fh1 = fh2 = None
    # Try/catch so can report
    try:
        fh1 = gzip.open(comargs.fq1, "rb")
        fh2 = gzip.open(comargs.fq2, "rb")
    except Exception as e:
        print("Problem opening fastq files")
        print(e)

    return fh1, fh2


def get_amp_part_bounds(seq):
    """Get start and end coords (for slice) for non-base parts in seq

    Assumes amplicon like below, with BC as digit, (optional) polyN as N

    Result lists are indexed [0]=N; [1]=round1; [2]=round2; [3]=round3

    Return arrays of start and end coords (for string slicing)
    """
    non_dna = set(seq.upper()) - DNA_BASES
    num_parts = len(non_dna)
    # If no N given, bump size
    if "N" not in non_dna:
        num_parts += 1
    part_ends = [-1] * num_parts
    part_starts = [-1] * num_parts
    # Scan each char; Annotate non-base parts
    prev_pi = -1
    i = 0
    while i < len(seq):
        pi = bc_amp_part_index(seq[i])
        # Change of part
        if pi != prev_pi:
            if pi >= 0:
                part_starts[pi] = i
            if prev_pi >= 0:
                part_ends[prev_pi] = i
        prev_pi = pi
        i += 1

    # Last end coord
    if part_ends[prev_pi] < 0:
        part_ends[prev_pi] = i
    return part_starts, part_ends


def bc_amp_part_index(b):
    """Part index for given character

    If normal DNA base, -1
    If 'N' then = 0 for polyN
    If int, then int N for barcode-N
    """
    if b in DNA_BASES:
        index = -1
    else:
        if b == "N":
            index = 0
        else:
            index = int(b)
    return index


def get_barcode_info(comargs, all_bc_dict):
    """Get barcode data for given kit

    all_bc_dict has data(frame) and (edit)dict for all available barcode sets
    comargs includes kit info

    Return bc_info dict with lists of bc data
    """
    bc_info = {}
    # Barcode sequences and error-correct edit-dist dicts loaded from files

    # Get (1-based) list of barcode set names based on kit
    bc_list = kit_bc_set_list(comargs.kit, comargs.chemistry)

    # Init lists with nothing (i.e. no round 0, rounds 1-3 have stuff)
    bc_seq_dfs = [None]
    bc_seqs = [None]
    bc_dicts = [None]
    # bc_seq_to_id_int = [None]
    # bc_seq_to_id_str = [None]
    bc_seq_to_wind = [None]
    num_wells = [0]

    bc_path = comargs.bcpath if comargs.bcpath else BC_DIR
    for i, bc in enumerate(bc_list):
        # First element has nothing
        #if i == 0:
        #    continue

        if bc not in all_bc_dict:
            print(f"Problem: barcode set '{bc}' not found in {bc_path}")
            return None

        # Edit-dist dict
        # Need to make sure edit dist keys are int (not str, as get from json decode)
        new_dict = {}
        for k, v in all_bc_dict[bc]["dict"].items():
            new_dict[int(k)] = v
        bc_dicts.append(new_dict)

        # Seq data dataframe; Add 'well_int' col
        bc_df = all_bc_dict[bc]["data"]
        bc_df = bc_df_add_well_int(bc_df)

        # Save dataframe but also sequences as list and well counts
        bc_seq_dfs.append(bc_df)
        bc_seqs.append(list(bc_df["sequence"].values))
        num_wells.append(len(bc_df["well"].unique()))

        map_dict = get_bc_data_dicts(bc_df)
        bc_seq_to_wind.append(map_dict["bc_seq_to_wind"])

    # Pack everything
    bc_info["bc_seq_dfs"] = bc_seq_dfs
    bc_info["bc_seqs"] = bc_seqs
    bc_info["num_wells"] = num_wells
    bc_info["bc_seq_to_wind"] = bc_seq_to_wind
    bc_info["bc_dicts"] = bc_dicts

    # Amplicon info; Seq and start,end for each round (zero = polyN)
    bc_info["amp_seq"] = comargs.amp_seq
    bc_starts, bc_ends = get_amp_part_bounds(comargs.amp_seq)
    bc_info["bc_starts"] = bc_starts
    bc_info["bc_ends"] = bc_ends

    return bc_info


def get_bc_data_dicts(df):
    """Get field-to-field mapping dicts for bc dataframes

    Return dict of dicts
    """
    # Sequence keys
    bc_seq_to_bci = dict(zip(df["sequence"].values, df.index.values))
    bc_seq_to_wind = dict(zip(df["sequence"].values, df["well_int"].values))
    bc_seq_to_well = dict(zip(df["sequence"].values, df["well"].values))
    bc_seq_to_type = dict(zip(df["sequence"].values, df["stype"].values))
    # String formatted well indexes
    bc_wind_to_str = get_bc_wind_str_dict(list(df["well_int"].values))
    bc_seq_to_wind_str = {k: bc_wind_to_str[v] for k, v in bc_seq_to_wind.items()}
    # bci keys
    bc_bci_to_wind = dict(zip(df.index.values, df["well_int"].values))
    bc_bci_to_well = dict(zip(df.index.values, df["well"].values))
    bc_bci_to_type = dict(zip(df.index.values, df["stype"].values))
    bc_bci_to_seq = dict(zip(df.index.values, df["sequence"].values))

    new_dict = {
        "bc_seq_to_bci": bc_seq_to_bci,
        "bc_seq_to_well": bc_seq_to_well,
        "bc_seq_to_type": bc_seq_to_type,
        "bc_seq_to_wind": bc_seq_to_wind,
        "bc_seq_to_wind_str": bc_seq_to_wind_str,
        "bc_wind_to_str": bc_wind_to_str,
        "bc_bci_to_wind": bc_bci_to_wind,
        "bc_bci_to_well": bc_bci_to_well,
        "bc_bci_to_type": bc_bci_to_type,
        "bc_bci_to_seq": bc_bci_to_seq,
    }
    return new_dict


def get_bc_wind_str_dict(bc_winds, pad=True, pad_to=0):
    """Get dict mapping well index (int) to str

    Return dict
    """
    assert isinstance(bc_winds, list), f"Expected list got {type(bc_winds)}"

    # Format string; Padded has leading zeros
    if pad_to:
        fmt = "{v:0" + str(pad_to) + "d}"
    elif pad:
        fmt = "{v:0" + str(len(str(max(bc_winds)))) + "d}"
    else:
        fmt = "{v}"
    new_dict = {k: fmt.format(v=k) for k in bc_winds}
    return new_dict


def well_to_wind_dicts():
    """Get dicts mapping between well string <--> 1-based int
    Assumes 96-well plate

    Return two dicts
    """
    well_to_wind = {}
    wind_to_well = {}
    n = 1
    for row in "ABCDEFGH":
        for c in range(12):
            col = f"{c+1}"
            well = row + col
            well_to_wind[well] = n
            wind_to_well[n] = well
            n += 1
    return well_to_wind, wind_to_well


def bc_df_add_well_int(df):
    """ """
    well_to_wind, _ = well_to_wind_dicts()
    df["well_int"] = df["well"].map(well_to_wind)
    return df


# ----------------------------- Group (well) specs ----------------------------
# Groups of wells for output
#
def get_outsets(comargs, bc_info):
    """Get output well/cell subset collection

    Return list of Outset objects, number of barcodes to check
    """
    # Collect subset defs; Sample wells or cells
    def_list = []
    # Samples will only look at first (one) barcode
    if comargs.gfile:
        def_list = load_def_list(comargs.gfile)
    elif comargs.group:
        for samp_def in comargs.group:
            if len(samp_def) < 2:
                print(f"Problem parsing '{samp_def}' as groups (wells)")
                def_list = []
                break
            def_list.append(samp_def[:2])

    # Should have something
    if not def_list:
        return None

    # Collect objects with parsed well/cell identifiers
    max_wells = bc_info["num_wells"][1]
    oset_list = []
    for name, s_def in def_list:
        bc_lis = parse_96wells(s_def)
        if not bc_lis:
            print(f"Problem parsing {name} '{s_def}' as groups (wells)")
            return None
        # Too many?
        if max(bc_lis) > max_wells:
            print(
                f"Problem parsing {name} '{s_def}'; Kit set for {max_wells} wells ({max(bc_lis)})"
            )
            return None

        oset_list.append(
            Outset(name, s_def, bc_lis, obase=comargs.obase, opath=comargs.opath)
        )

    return oset_list


def load_def_list(fname):
    def_list = []
    with open(fname) as INFILE:
        for line in INFILE:
            # Ignore '#' commented out / blank lines
            parts = line.strip().split("#")[0].split()
            # Def should have <name> <wells/cells>
            if len(parts) >= 2:
                def_list.append(parts[:2])
    return def_list


# Well definition regex match patterns
# Single well, Colon:range, Dash-range; Get letter and number separate
WELL_1_REGEX = re.compile(r"([ABCDEFGH])([1-9]|1[012])$")
WELL_C_REGEX = re.compile(r"([ABCDEFGH])([1-9]|1[012]):([ABCDEFGH])([1-9]|1[012])$")
WELL_D_REGEX = re.compile(r"([ABCDEFGH])([1-9]|1[012])-([ABCDEFGH])([1-9]|1[012])$")


def parse_96wells(s):
    """Parse well specification string into well indexes

    Return list of well indexes; As 1-based int
    """
    wells = np.arange(96, dtype=int).reshape(8, 12)
    row_letter_to_number = dict(zip(list("ABCDEFGH"), [i for i in range(8)]))
    sub_wells = []
    # Try will fail on non-match that is processed regardless
    try:
        blocks = s.upper().split(",")
        for b in blocks:
            if ":" in b:
                vals = unpack_well_match(WELL_C_REGEX.match(b), 4)
                s_row = row_letter_to_number[vals[0]]
                s_col = vals[1] - 1
                e_row = row_letter_to_number[vals[2]]
                e_col = vals[3]
                sub_wells += list(wells[s_row : e_row + 1, s_col:e_col].flatten())
            elif "-" in b:
                vals = unpack_well_match(WELL_D_REGEX.match(b), 4)
                s_row = row_letter_to_number[vals[0]]
                s_col = vals[1] - 1
                e_row = row_letter_to_number[vals[2]]
                e_col = vals[3] - 1
                sub_wells += list(
                    np.arange(wells[s_row, s_col], wells[e_row, e_col] + 1)
                )
            else:
                vals = unpack_well_match(WELL_1_REGEX.match(b), 2)
                s_row = row_letter_to_number[vals[0]]
                s_col = vals[1] - 1
                sub_wells += [wells[s_row, s_col]]
        sub_wells = list(np.unique(sub_wells))
    except Exception:
        # print(e)
        pass

    # Convert to 1-base
    sub_wells = [i + 1 for i in sub_wells]
    return sub_wells


def unpack_well_match(mat, num):
    """Unpack well regex match with two or four items [letter, well, [letter, well]]"""
    vals = []
    if num == 2:
        vals = [mat[1], int(mat[2])]
    elif num == 4:
        vals = [mat[1], int(mat[2]), mat[3], int(mat[4])]
    return vals


# Class to hold info on output groups of wells
class Outset(object):
    def __init__(self, name, sdef, id_list, obase="", opath=""):
        self.name = name
        self.sdef = sdef
        self.count = 0
        self.set_fnames(obase, opath)
        self.fq1 = None
        self.fq2 = None
        # Set of indices to match
        self.iset = set(id_list)
        # Output file buffers
        self.init_buffers()

    def __del__(self):
        self.close_ofiles()

    def __repr__(self):
        ostring = "Outset (output set definition)\n"
        ostring += f"Name:    {self.name}\n"
        ostring += f"SampDef: {self.sdef}\n"
        ostring += f"Set:     {len(self.iset)}: {self.iset}\n"
        ostring += f"Files:   {self.fq1_fname}   {self.fq2_fname}\n"
        return ostring

    def set_fnames(self, obase="", opath=""):
        """Set output fastq filenames"""
        name = self.get_name()
        # If base name (prefix), add underscore to sep from name
        if obase:
            obase = obase + "_"
        if opath:
            if not opath.endswith("/"):
                opath = opath + "/"
        self.fq1_fname = f"{opath}{obase}{name}_R1.fastq.gz"
        self.fq2_fname = f"{opath}{obase}{name}_R2.fastq.gz"

    def get_fname(self, fq1=True, fq2=False):
        """Return one or more output filenames"""
        fname = ""
        if fq1 and fq2:
            fname = (self.fq1_fname, self.fq2_fname)
        elif fq1:
            fname = self.fq1_fname
        elif fq2:
            fname = self.fq2_fname
        return fname

    def get_ofiles(self):
        """Get output (gzip fastq) files for this sample; Open if needed"""
        if self.fq1 is None:
            self.fq1 = gzip.open(self.fq1_fname, "wb", compresslevel=GZIP_COMPRESSLEVEL)
            self.fq2 = gzip.open(self.fq2_fname, "wb", compresslevel=GZIP_COMPRESSLEVEL)
        return self.fq1, self.fq2

    def close_ofiles(self, verb=True, both=False):
        self.write_out_bufs()
        if self.fq1:
            if verb:
                n = self.get_count()
                if both:
                    print(f"New files: {self.fq1.name}  {self.fq2.name}  {n}")
                else:
                    print(f"New files: {self.fq1.name}  (and R2)  {n}")
            self.fq1.close()
            self.fq2.close()
            self.fq1 = self.fq2 = None

    def init_buffers(self):
        self.r1_buf = b""
        self.r2_buf = b""
        self.n_buf = 0

    def write_rec(self, head1, seq1, plus1, qual1, head2, seq2, plus2, qual2):
        """Handle writing for two fastq records; Four lines each R1 R2"""
        self.r1_buf += head1 + seq1 + plus1 + qual1
        self.r2_buf += head2 + seq2 + plus2 + qual2
        self.n_buf += 1
        if self.n_buf >= GZIP_REC_BUFFER:
            self.write_out_bufs()
        # TODO; Better error handling
        return True

    def write_out_bufs(self):
        """Write any bufferd lines to file"""
        if self.r1_buf:
            ofile1, ofile2 = self.get_ofiles()
            ofile1.write(self.r1_buf)
            ofile2.write(self.r2_buf)
            # Reset buffer
            self.init_buffers()

    def inc_count(self, inc=1):
        self.count += inc

    def get_count(self):
        return self.count

    def get_iset(self):
        return self.iset

    def get_name(self):
        return self.name

    def get_sampdef(self, out_set=True):
        ostr = self.sdef
        if out_set:
            slist = sorted([int(i) for i in self.iset])
            ostr += f"\t({len(slist)}):\t{slist}"
        return ostr

    def get_report(self):
        ostring = f"{self.get_name()}\t{self.get_sampdef()}"
        return ostring


def report_settings(comargs, bc_info, out_sets):
    """Report story"""
    print(f"# Fasta R1      {comargs.fq1}")
    print(f"#       R2      {comargs.fq2}")
    story = range_story(comargs.range)
    print(f"# Read range    {story}")
    print(f"# Kit           {comargs.kit}")
    kit_list = kit_bc_set_list(comargs.kit, comargs.chemistry)
    print(f"# Barcodes      {kit_list}")
    print(f"# Amplicon      {bc_info['amp_seq']}")
    print(f"# Max edit dist {comargs.edit_dist}")
    print("# Matching      Permissive first barcode matches")
    bc_path = comargs.bcpath if comargs.bcpath else BC_DIR
    print(f"# Barcode path  {bc_path}")
    print(f"# Out path      {comargs.opath}")
    print(f"# Out base      {comargs.obase}")
    print("#")
    if comargs.gfile:
        print(f"# Group file    {comargs.gfile}")
    print(f"# Subsets       {len(out_sets)} groups (by wells)")
    for oset in out_sets:
        print(f"#   {oset.get_report()}")
    print("#")
    if not comargs.only_stats:
        print("# Output files")
        for oset in out_sets:
            name = oset.get_name()
            fname = oset.get_fname()
            print(f"#   {name}\t{fname}")
        print("#")


def add_Q30_stats(read_stats, fq2_df):
    """Add Q30 stats

    Return (updated given) dict
    """
    read_stats["bc1_Q30"] = round(np.mean(fq2_df["qc1"].apply(seq_qual_score)), 3)
    read_stats["bc2_Q30"] = round(np.mean(fq2_df["qc2"].apply(seq_qual_score)), 3)
    read_stats["bc3_Q30"] = round(np.mean(fq2_df["qc3"].apply(seq_qual_score)), 3)
    return read_stats


def seq_qual_score(qual):
    """Get Q30 score for list of quality srings"""
    # Convert seq quality string into mean Q score
    # https://www.biostars.org/p/9463767/
    #
    # Pipeline = mean fraction "good" (Q30) bases
    m = np.mean([ord(c) > 62 for c in qual])
    # m = np.mean([ord(c) - 33 for c in qual])

    return m


def get_min_edit_dists(bc, edit_dict, max_d):
    """Returns a list of nearest edit dist seqs
    Input 8nt barcode, edit_dist_dictionary
    Output <list of nearest edit distance seqs>, <edit dist>
    """
    bc_matches = edit_dict[0].get(bc, [])
    edit_dist = 0
    found = bool(bc_matches)
    while (not found) and (edit_dist < max_d):
        edit_dist += 1
        bc_matches = edit_dict[edit_dist].get(bc, [])
        if bc_matches:
            found = True
            break
        if edit_dist >= max_d:
            break
    return bc_matches, edit_dist, found


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
