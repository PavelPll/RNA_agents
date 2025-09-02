# Read and Write FASTA files

def read_single_fasta(filepath):
    """Reads a single sequence from a FASTA file.

    Returns:
        tuple: (header, sequence)
    """
    header = None
    sequence = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                header = line[1:]  # Remove '>'
            else:
                sequence.append(line)

    if header is None:
        raise ValueError("No FASTA header found in file.")

    return header, ''.join(sequence)


def write_single_fasta(header, sequence, filepath):
    """Writes a single sequence to a FASTA file.

    Args:
        header (str): The FASTA header (no '>')
        sequence (str): The sequence
        filepath (str): Output file path
    """
    with open(filepath, 'w') as f:
        f.write(f">{header}\n")
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i+80] + "\n")