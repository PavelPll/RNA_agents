# RNA_agents
LLM Agent–Driven ncRNA Design via Intrinsic Features and Structure-Guided Feedback

## Description 
* Construction of Tools for RNA sequence analysis:
    * From reconstructed RNA 3D structure
    * From RNA conditional diffusion model
    * From internet search
    * From PubMed abstracts, followed by full-text paper search on the internet
* Tool-Calling Agent implementation
* ReAct Agent implementation
* For a quick presentation of the results and additional implementation details, click [here](https://github.com/PavelPll/RNA_agents/blob/main/docs/rna_agents.pdf)

## Getting Started

### Dependencies
* Large Language Models ([Claude Sonnet 4 (20250514) Anthropic.](https://www.anthropic.com)
* LangGraph (2024): [Low-level orchestration framework](https://github.com/langchain-ai/langgraph) for building, managing, and deploying long-running, stateful agents 
* RiboDiffusion model to get FASTA (RNA sequence) from PDB (3D structure) ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11211841/), [GitHub](https://github.com/ml4bio/RiboDiffusion))
* DRfold2 model to get PDB (3D structure) from FASTA (RNA sequence) ([paper](https://www.biorxiv.org/content/10.1101/2025.03.05.641632v1), [GitHub](https://github.com/leeyang/DRfold2.git))
* [DSSR](http://skmatic.x3dna.org/) to extract RNA properties from its 3D structure (PDB file) [paper](https://academic.oup.com/nar/article/48/13/e74/5842193?login=false)
* [RNA-FM](https://huggingface.co/multimolecule/rnafm): Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions
* [RNAcentral DATABASE](https://rnacentral.org) of non-coding RNA (ncRNA) sequences
* [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) to get some physicochemical properties
* [ViennaRNA](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html) predicting and comparing RNA secondary structures
* [IUPAC code](https://www.bioinformatics.org/sms/iupac.html) for nucleotides and amino acids
* Windows 11, Visual Studio Code
* Docker
* Torch

### Installing

I adapted the same conda environment for both LLM agents and RiboDiffusion model. However, I installed DRfold2 in a Docker container running Ubuntu 22.04 because ARENA package requires Linux for compilation (see RNA_agents/Dockerfile). The Large Language Model (LLM) requires the key, please get it [here](https://www.anthropic.com). I use an NVIDIA GeForce RTX 4060 with 8 GB VRAM and 32 GB of RAM to run DRfold2 and Ribodiffusion models. A single simulation step takes about 5–10 minutes.
* Clone the repository::
```
git clone https://github.com/PavelPll/RNA_agents.git
cd RNA_agents
```
* Install DRfold2 inside a Docker container:
```
cd RNA_agents
git clone https://github.com/leeyang/DRfold2.git drfold2
git clone https://github.com/pylelab/Arena.git drfold2/Arena
cd drfold2
mkdir file_exchange\fasta_input && mkdir file_exchange\pdb_output
docker build -t drfold_image ../
docker run --gpus all -it --name drfold_container -v .:/opt/drfold2 drfold_image bash
Run inside container:
wget --header="User-Agent: Mozilla/5.0" https://zhanglab.comp.nus.edu.sg/DRfold2/res/model_hub.tar.gz
tar -xzvf model_hub.tar.gz
rm -rf model_hub.tar.gz
cd Arena
make Arena
exit
Go back to RNA_RAG folder:
cd ..
```

* Install RiboDiffusion:
```
cd RNA_agents
git clone https://github.com/ml4bio/RiboDiffusion
cd RiboDiffusion
Model checkpoint can be downloaded from here. 
https://drive.google.com/drive/folders/10BNyCNjxGDJ4rEze9yPGPDXa73iu1skx
Another checkpoint trained on the full dataset (with extra 0.1 Gaussian noise for coordinates) can be downloaded from here.
https://drive.google.com/file/d/1-IfWkLa5asu4SeeZAQ09oWm4KlpBMPmq/view
Download and put the checkpoint files in the RiboDiffusion/ckpts folder.
```
* Set up a conda environment:
```
conda create -n rna_agents python=3.11.11
conda activate rna_agents

pip install -q torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install requests matplotlib
pip install transformers sentence-transformers langchain langchain-community langchain_anthropic
pip install torch_geometric==2.3.1 torch_scatter==2.1.1 torch_cluster==1.6.1
pip install fair_esm==2.0.0 ml_collections==0.1.1
conda install -c conda-forge dm-tree=0.1.7
pip install biopython==1.80
pip install -U ddgs
pip install wikipedia
pip install easy-entrez
pip install langgraph
```
* Install ViennaRNA from [here](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/install.html)

### Executing program
* ####  Large Language Model (LLM) ReAct Agent–Driven RNA Design via Structural Feedback and RNA Diffusion Model
    ```
   notebooks/RNA_reactAgent.ipynb
    ```
    * Comparison of two RNAs
    * Generate two connected RNA hairpins without RNA template
    * Generate two connected RNA hairpins from a given RNA template
    * Modification of a specified region of an RNA molecule
* #### An attempt to model RNA evolution using (LLM) tool agent
     ```
     notebooks/RNA_toolAgent.ipynb
     ```
    * Model input: FASTA file with initial RNA sequence (e.g. trnaGlycine_Asgard_group_archaeon.fasta from data/processed/rna_evolution_seed folder;
    * Model output: ancestral_sequence in FASTA and PDB formats with detailed description of each evolution step (evolution_steps.txt) in data/processed/rna_evolution folder.

## License
This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details



> [!NOTE]
> For more information see short [presentation](https://github.com/PavelPll/RNA_agents/blob/main/docs/rna_agents.pdf)

