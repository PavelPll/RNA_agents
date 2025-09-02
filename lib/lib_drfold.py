# I implemented DRfold2 in a Docker container
# Implemented from here: https://github.com/leeyang/DRfold2.git 

import subprocess
import shutil
import os
import sys
sys.path.append(os.path.abspath('../'))

def cpy_file(source_path, destination_path):
    # Copy file to and from the Docker container
    # Make sure the destination directory exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    # Copy and rename
    shutil.copy2(source_path, destination_path)
    print(f"Copied and renamed to: {destination_path}")

def clear_output(output_dir = "../drfold2/file_exchange/pdb_output"):
    # Delete existing output in the container before running
    # Loop through all contents of the folder
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # delete file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # delete folder and its contents
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

def run_container():
    clear_output()
    # Define your Docker container name or ID
    container_name = "drfold_container"

    # Define the command to run inside the container
    command = """
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate drfold && \
    python DRfold_infer.py ./file_exchange/fasta_input/fasta.fasta ./file_exchange/pdb_output
    """

    # Build the docker exec command
    docker_command = ["docker", "exec", container_name, "bash", "-c", command]

    # Run the command and capture output
    try:
        result = subprocess.run(docker_command, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)

def fasta2pdb(source_path):
    # GET PDB file from the FASTA file
    # Copy to container
    cpy_file(source_path=source_path, destination_path="../drfold2/file_exchange/fasta_input/fasta.fasta")
    run_container()

    destination_path= source_path.replace('.fasta', '.pdb')
    # Copy from container
    cpy_file(source_path = "../drfold2/file_exchange/pdb_output/relax/model_1.pdb", destination_path=destination_path)
    return destination_path