from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
#from langchain_core.tools import Tool
from langchain.tools import Tool
from datetime import datetime
import json
from lib.lib_x3dna import format_rna_structure
from Bio import pairwise2
from datetime import datetime
from lib.fasta_utils import read_single_fasta, write_single_fasta
from lib.lib_drfold import fasta2pdb
from lib.lib_ribodiffusion import Pdb2Fasta
from lib.lib_x3dna import RnaProperties
import os
import yaml

# Read config
with open('../configs/rna_rag.yaml', 'r') as f:
    config = yaml.safe_load(f)
    input_dir = config['input_output']['input_dir']
    output_dir = config['input_output']['output_dir_ReActAgent']

#-----------------------------------------------------------------------------
# Get additional information from sequence and save it to .json file
# .json file is acessible by different tools
def sequence2information(sequence: str, output_dir=output_dir):
    # Get additional information from sequence and save it to .json file
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    RNAreport = output_dir + f"RNAreport_{timestamp}"

    #sequence = read_single_fasta(RNAreport+".fasta")[1]
    write_single_fasta(header="seq", sequence=sequence, filepath=RNAreport+".fasta")

    # print("- Calculating PDB from FASTA") 
    pdb_path = fasta2pdb(source_path = RNAreport+".fasta")

    # print("- Calculating RNA properties from its 3D structure")
    RNAproperties_3D = RnaProperties(pdb_input=pdb_path)
    genInfo = RNAproperties_3D.get_genInfo()
    helicesInfo = RNAproperties_3D.get_helicesInfo()
    helicesInfo = helicesInfo if isinstance(helicesInfo, dict) else {}
    stemsInfo = RNAproperties_3D.get_stemsInfo()
    stemsInfo = stemsInfo if isinstance(stemsInfo, dict) else {}

    print()
    # print("- Calculating backfolded_rna")
    pdb2fasta = Pdb2Fasta()
    backfolded_sequences = pdb2fasta.pdb2fasta(source_path = pdb_path, cond_scale=-1.)
    backfolded_sequences = sorted(backfolded_sequences, key=lambda x: x[1], reverse=True) # sort by similarity to current_sequence
    # print(len(backfolded_sequences))
    backfolded_sequence, similarity_score = backfolded_sequences[0]
    genInfo["backfolded_sequence"] = backfolded_sequence
    genInfo["sequence"] = sequence
    genInfo["helices"] = helicesInfo
    genInfo["stems"] = stemsInfo
    dssrInfo = genInfo
    # dssrInfo = (genInfo | helicesInfo)

    # Save to a file
    with open(RNAreport+".json", "w") as f:
        json.dump(dssrInfo, f, indent=4)  # indent is optional (for pretty formatting)
    return dssrInfo

def get_dssrInfo(current_rna_sequence: str, output_dir=output_dir):

    json_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
    dssrInfo = {}
    for json_file in json_files:
        with open(output_dir+json_file, "r") as f:
            dssrInfo_ = json.load(f)
        if dssrInfo_["sequence"]==current_rna_sequence:
            dssrInfo = dssrInfo_
            break
    if not dssrInfo:
        dssrInfo = sequence2information(current_rna_sequence)
    return dssrInfo

def get_hairpins(current_rna_sequence: str):
    dssrInfo = get_dssrInfo(current_rna_sequence)
    out = ", ".join(dssrInfo["hairpins"]) if dssrInfo.get("hairpins") else "No hairpins found"
    return out
get_hairpins_tool = Tool(
    name="get_hairpins",
    func=get_hairpins,
    description = "Identify all hairpin loops and provide their sequences.",
)

def get_helices(current_rna_sequence: str):
    dssrInfo = get_dssrInfo(current_rna_sequence)
    return format_rna_structure(dssrInfo.get("helices", "No helices found in the RNA structure"))
get_helices_tool = Tool(
    name="get_helices",
    func=get_helices,
    description = "Extract all helices from this RNA structure, including base pairs, strand sequences, and helix form",
)

def get_stems(current_rna_sequence: str):
    dssrInfo = get_dssrInfo(current_rna_sequence)
    return format_rna_structure(dssrInfo.get("stems", "No stems found in the RNA structure"))
get_stems_tool = Tool(
    name="get_stems",
    func=get_stems,
    description = "Extract all stems from this RNA structure, including base pairs, strand sequences, and helix form",
)

def get_dotBracket_length(current_rna_sequence: str):
    dssrInfo = get_dssrInfo(current_rna_sequence)
    response1 = "2D RNA structure in dot-bracket notation: " + str(dssrInfo['dot_bracket'][0])
    response2 = "RNA length: " + str(len(current_rna_sequence)) + " nucleotides"
    response = response1 + " " + response2
    return response
get_dotBracket_length_tool = Tool(
    name="get_dotBracket_length",
    func=get_dotBracket_length,
    description = "Returns the RNA secondary structure in dot-bracket notation, representing base pairing "
        "and loop regions, along with the total RNA sequence length.",
)

def get_multiplets(current_rna_sequence: str):
    dssrInfo = get_dssrInfo(current_rna_sequence)
    out = ", ".join(dssrInfo["multiplets"]) if dssrInfo.get("multiplets") else "No multiplets found"
    return out
get_multiplets_tool = Tool(
    name="get_multiplets",
    func=get_multiplets,
    description = "Identifies multiplets—clusters of three or more interacting nucleotides—in the RNA 3D structure.",
)    

def get_junctions(current_rna_sequence: str):
    dssrInfo = get_dssrInfo(current_rna_sequence)
    out = ", ".join(dssrInfo["junctions"]) if dssrInfo.get("junctions") else "No junctions found"
    return out
get_junctions_tool = Tool(
    name="get_junctions",
    func=get_junctions,
    description = "Identifies junction sequences where multiple RNA helices meet, forming multi-branch loop regions critical for RNA folding.",
) 

def get_riboseZippers(current_rna_sequence: str):
    dssrInfo = get_dssrInfo(current_rna_sequence)
    out = ", ".join(dssrInfo["riboseZippers"]) if dssrInfo.get("riboseZippers") else "No riboseZippers found"
    return out
get_riboseZippers_tool = Tool(
    name="get_riboseZippers",
    func=get_riboseZippers,
    description = "Identifies ribose zippers—motifs formed by consecutive hydrogen bonds between ribose 2'-OH groups and nearby bases.",
) 


def get_stacks(current_rna_sequence: str):
    dssrInfo = get_dssrInfo(current_rna_sequence)
    out = ", ".join(dssrInfo["stacks"]) if dssrInfo.get("stacks") else "No stacks found"
    return out
get_stacks_tool = Tool(
    name="get_stacks",
    func=get_stacks,
    description = "Identifies base stacking interactions between adjacent nucleotides in the RNA 3D structure.",
) 
def get_nonStack(current_rna_sequence: str):
    dssrInfo = get_dssrInfo(current_rna_sequence)
    out = ", ".join(dssrInfo["nonStack"]) if dssrInfo.get("nonStack") else "No nonStack found"
    return out
get_nonStack_tool = Tool(
    name="get_nonStack",
    func=get_nonStack,
    description = "Identifies nucleotides that are not involved in any base stacking interactions.",
) 
def get_numCoaxStacks(current_rna_sequence: str):
    dssrInfo = get_dssrInfo(current_rna_sequence)
    return dssrInfo.get("num_coaxStacks", "No coaxStacks found")
get_numCoaxStacks_tool = Tool(
    name="get_numCoaxStacks",
    func=get_numCoaxStacks,
    description = "Identifies a number of coaxial stacking between RNA helices—where multiple stems align along a common helical axis.",
)


"""def get_stacks_nonStack_numCoaxStacks(current_rna_sequence: str):
    # combine 3 tools to single one ... 
    dssrInfo = get_dssrInfo(current_rna_sequence)
    response1 = "Base stacking interactions between adjacent nucleotides: " + ", ".join(dssrInfo["stacks"]) if dssrInfo.get("stacks") else "No stacks found"
    response2 = "Nucleotides that are not involved in any base stacking interactions: " + ", ".join(dssrInfo["nonStack"]) if dssrInfo.get("nonStack") else "No nonStack found"
    response3 = " number of coaxial stacking between RNA helices—where multiple stems align along a common helical axis: " + dssrInfo.get("num_coaxStacks", "No coaxStacks found")
    response = response1 + ", " + response2 + ", " + response3 + "."
    return response
get_stacks_nonStack_numCoaxStacks_tool = Tool(
    name="get_get_stacks_nonStack_numCoaxStacks",
    func=get_stacks_nonStack_numCoaxStacks,
    description = "Characterizes stacking and non-stacking interactions, and reports the number of coaxial stacks in the RNA.",
)"""

def get_length_pairs_hydrogenBonds(current_rna_sequence: str):
    dssrInfo = get_dssrInfo(current_rna_sequence)
    response1 = "Length: " + str(len(current_rna_sequence)) + " nucleotides"
    response2 = "Number of base pairs: " + str(dssrInfo.get('base_pairs', "No pairs found in the RNA structure"))
    response3 = "Number of hydrogen bonds: " + str(dssrInfo.get('hydrogen_bonds', "No hydrogen bonds found in the RNA structure"))
    response = response1 + ", " + response2 + ", " + response3 + "."
    return response
get_length_pairs_hydrogenBonds_tool = Tool(
    name="get_get_length_pairs_hydrogenBonds",
    func=get_length_pairs_hydrogenBonds,
    description = "Returns number of hydrogen bonds, number of base pairs and number of hydrogen bonds in this RNA.",
)

def get_backfoldedRNAsequence(current_rna_sequence: str):
    dssrInfo = get_dssrInfo(current_rna_sequence)
    return dssrInfo["backfolded_sequence"]
get_backfoldedRNAsequence_tool = Tool(
    name="get_backfoldedRNAsequence",
    func=get_backfoldedRNAsequence,
    description="Returns a backfolded RNA sequence that preserves the 3D structure of the input RNA sequence",
)

#-----------------------------------------------------------------------------
# Sequence Alignment TOOL
def alignment_to_json(alignment, score=None):
    """
    Convert BioPython alignment (pairwise or multiple) to JSON-friendly dict.
    Works with Bio.Align.MultipleSeqAlignment and pairwise2 results.
    """

    # Handle pairwise2.Alignment
    if hasattr(alignment, "seqA") and hasattr(alignment, "seqB"):
        sequences = [
            {"id": "target", "sequence": alignment.seqA},
            {"id": "query", "sequence": alignment.seqB},
        ]
        aligned_seqs = [alignment.seqA, alignment.seqB]
        aln_score = alignment.score if hasattr(alignment, "score") else score

    # Handle MultipleSeqAlignment
    elif isinstance(alignment, MultipleSeqAlignment):
        sequences = [{"id": rec.id, "sequence": str(rec.seq)} for rec in alignment]
        aligned_seqs = [str(rec.seq) for rec in alignment]
        aln_score = score

    else:
        raise TypeError("Unsupported alignment type")

    # Basic stats
    aln_len = len(aligned_seqs[0])
    num_seqs = len(aligned_seqs)
    gaps = [seq.count("-") for seq in aligned_seqs]

    # Column-wise comparison
    matches = 0
    mismatches = 0
    column_data = []
    identical_counts = {}

    for i in range(aln_len):
        column = [seq[i] for seq in aligned_seqs]
        column_data.append(column)
        
        # Only compare pairwise for now
        if num_seqs == 2:
            a, b = column
            if a == b and a != "-":
                matches += 1
                identical_counts[f"{a}-{b}"] = identical_counts.get(f"{a}-{b}", 0) + 1
            elif a != "-" and b != "-":
                mismatches += 1

    identity_pct = (matches / aln_len) * 100

    result = {
        "alignment_info": {
            "num_sequences": num_seqs, #2
            "alignment_length": aln_len,
            "gaps_per_sequence": {sequences[i]["id"]: gaps[i] for i in range(num_seqs)},
            "matches": matches,
            "mismatches": mismatches,
            "identity_percentage": round(identity_pct, 2),
        },
        "sequences": sequences,
        "aligned_sequences": aligned_seqs,
        "score": aln_score,
        "identical_nucleotide_counts": identical_counts,
        "alignment_columns": column_data
    }
    
    return result

def describe_alignment_for_llm(alignment_data: dict) -> str:
    info = alignment_data["alignment_info"]
    score = alignment_data["score"]
    identity_map = alignment_data.get("identical_nucleotide_counts", {})
    sequences = alignment_data.get("sequences", [])

    output = []
    output.append("### Alignment Summary\n")
    output.append(f"- Number of sequences: **{info['num_sequences']}**")
    output.append(f"- Alignment length: **{info['alignment_length']} nucleotides**")
    output.append(f"- Alignment score: **{score}**")
    output.append(f"- Matches: **{info['matches']}**, Mismatches: **{info['mismatches']}**")
    output.append(f"- Identity: **{info['identity_percentage']}%**")
    output.append("\n### Gaps per Sequence")
    for seq_id, gap_count in info["gaps_per_sequence"].items():
        output.append(f"- `{seq_id}`: {gap_count} gaps")

    if identity_map:
        output.append("\n### Identical Nucleotide Matches")
        for pair, count in identity_map.items():
            output.append(f"- {pair}: {count}")

    if sequences:
        output.append("\n### Aligned Sequences")
        for seq in sequences:
            output.append(f"- **{seq['id']}**: `{seq['sequence']}`")

    return "\n".join(output)

def RNA_2alignment(sequence1, input_dir=input_dir):
    fasta_file = input_dir + "trnaGlycine_Asgard_group_archaeon.fasta"
    sequence2 = read_single_fasta(fasta_file)[1]

    alignments = pairwise2.align.globalxx(sequence2, sequence1)
    al_json = alignment_to_json(alignments[0])
    return describe_alignment_for_llm(al_json)

get_RNA_2alignment_tool = Tool(
    name="RNA_2alignment",
    func=RNA_2alignment,
    description=(
        "Aligns the input RNA sequence (query) with a fixed reference RNA sequence (target) "
        "given by GCACCGAUAGUCUAGUGGUAGGACUUAUCCCUGCCUAGGAUAGAGCCCGGGUUCAAAUCCCGGUCGGUGCACCA ."
        "Returns the alignment score, identity percentage, matches, gaps, and the aligned sequences."
    )
)