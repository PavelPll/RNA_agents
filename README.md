# RNA_agents
Using LLM agents to Decode and Evolve RNA Molecules

## Description 
* Context construction:
    * from RNA 3D structure
    * from RNA diffusion model
* Tool_calling_agent implementation
* For more information click [here](https://github.com/PavelPll/RNA_agents/blob/main/docs/rna_agents.pdf)



## Getting Started

### Dependencies
* Large Language Models ([Claude Sonnet 4 (20250514) Anthropic.](https://www.anthropic.com)
* Text-visual transformers (clip-ViT-B-32, EVA02-E-14-plus, ...)
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

I adapted the same conda environment for both RAG and RiboDiffusion. However, I installed DRfold2 in a Docker container running Ubuntu 22.04 because the ARENA package requires Linux for compilation (see RNA_RAG/Dockerfile). The Large Language Model (LLM) is installed locally, so no specific keys are needed. I use an NVIDIA GeForce RTX 4060 with 8 GB VRAM and 32 GB of RAM.
* To install the RAG system for RNA queries, clone the repository::
```
git clone https://github.com/PavelPll/RNA_RAG.git
cd RNA_RAG
```
* Install DRfold2:
```
cd RNA_RAG
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
cd RNA_RAG
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
conda create -n rna_rag2 python=3.11.11

pip install -q torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers matplotlib PyMuPDF sentence-transformers langchain langchain-community sentencepiece protobuf accelerate open-clip-torch optimum
conda install -c conda-forge faiss-gpu
pip uninstall numpy -y
pip install numpy==1.26.2
pip install bitsandbytes
pip install --no-deps xformers
pip install biopython==1.80
conda install -c conda-forge pymol-open-source
pip install torch_geometric==2.3.1 torch_scatter==2.1.1 torch_cluster==1.6.1
pip install fair_esm==2.0.0 ml_collections==0.1.1
conda install -c conda-forge dm-tree=0.1.7
pip install rna-fm
pip install matplotlib
pip install adjustText
```
* Install ViennaRNA from [here](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/install.html)
* Add RNA textbooks in PDF format to data/raw/books.

### Executing program

* #### Query the RNA of interest using a prompt
     ```
     notebooks/RAG_deployment.ipynb
     ```
     **Only two kinds of prompts are possible: general and evolutionary.** This is configured using the prompt_choice parameter in configs/rna_rag.yaml. The general prompt (prompt_choice = 0) is used to ask a question about a given RNA, while the evolutionary prompt (prompt_choice = 1) is used to generate a new RNA sequence. Both prompts can be visualized and modified, if needed, in lib/RAG_Biomistral/rag_templates.py. The corresponding context can be adjusted using other parameters in configs/rna_rag.yaml:
    * Choose a model to generate text and image embeddings using text_visual_encoder_choice:
         ```
         2 for sentence-transformers/clip-ViT-B-32 (on gpu) the best!
         3 for ViT-H-14 (on cpu because of long embedding) OK                  
         4 for EVA02-E-14-plus (on cpu because of long embedding) OK                      
         5 for hf-hub:imageomics/bioclip (on cpu because of long embedding) OK
         ```
    * Change text and image context configuration for prompt construction.:
        ```
        num_question_context_pages: 3 # context size in prompt
        num_image_context_pages: 3 # context size in prompt per image
        book_tokens_per_page: 200 # tokens per page of textbook (pdf)
        chunk_overlap_ratio: 0.1  
        ```
    * Adjust LLM performance quality and processing speed by selecting the degree of quantization used during inference and the model type:
        ```
        inference_8_bit: True # False for 4 bit; True for 8 bit
        llm_model_choice: 0 # 0 for Biomistral (recommended); 1 for medalpaca-7b (context is shorter)
        ```
* #### An attempt to model RNA evolution using a large language model (LLM)
     ```
     notebooks/RNA_evolution.ipynb
     ```
    * Model input: FASTA file with initial RNA sequence;
    * Model output: new FASTA, PDB, and PNG files of the 3D structure for each RNA generated at every simulation step.
## License
This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details



> [!NOTE]
> For more information see short [presentation](https://github.com/PavelPll/RNA_RAG/blob/main/docs/rna_rag.pdf)

