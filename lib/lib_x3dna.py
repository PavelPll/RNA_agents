# from here: http://skmatic.x3dna.org/

import requests
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io


def format_rna_structure(rna_dict):
    # Rewrite RNA structure in a more friendy for LLM way
    lines = []

    def flatten(key, value):
        if isinstance(value, dict):
            flat_items = []
            for subkey, subval in value.items():
                flat_items.append(f"{subkey}={subval}")
            return f"{key}: " + ', '.join(flat_items)
        elif isinstance(value, list):
            return f"{key}: " + ', '.join(map(str, value))
        else:
            return f"{key}: {value}"

    for key, value in rna_dict.items():
        lines.append(flatten(key, value))

    return '\n'.join(lines)

class RnaProperties():
    # Get RNA structure description from PDB file
    def __init__(self, pdb_input, sequence="sequence"):
        url = 'http://skmatic.x3dna.org/api'
        # pdb_file_path = 'results/2_73_-1_tRNA/fasta73_tRNA0.pdb'
        self.sequence = sequence
        self.pdb_input = pdb_input
        # print("--pdb_input", pdb_input)
        with open(pdb_input, 'rb') as pdb_file:
            files = {'model': pdb_file}
            data = {'type': 'json'}

            response = requests.post(url, files=files, data=data)

            if response.status_code == 200:
                try:
                    self.json_data = response.json()
                    #print(json.dumps(json_data, indent=2))
                except ValueError:
                    print("Response is not in JSON format.")
            else:
                print(f"Request failed with status code {response.status_code}")
        # print(self.json_data["multiplets"])

    def get_pdbImage(self):

        url = 'http://skmatic.x3dna.org/api'

        with open(self.pdb_input, 'rb') as pdb_file:
            files = {'model': pdb_file}
            data = {'block_file':'wc-minor'}     # wc-minor style
            #data = {}

            response = requests.post(url, files=files, data=data)

        # Simulated: binary_image_data = response.content
        binary_image_data = response.content  # your binary PNG bytes
        self.pdb_image = mpimg.imread(io.BytesIO(binary_image_data), format='png')

        # Display the image
        #plt.imshow(self.pdb_image)
        #plt.axis('off')  # Hide axis for a cleaner view
        #plt.show()
        return self.pdb_image




    # # All stems are helices, but not all helices are stems.
    def summarize_stems_minimal(self, helices, structure):
        """
        Generate an LLM prompt that asks for a concise summary of helix features.
        """
        # "avg_rise": h.get("helical_rise"), # 2.77 3.025
        # "radius" : h.get('helical_radius'),

        output = ""
        output_dict = {}
        ## output_dict["full_RNA_sequence"] = self.sequence
        for i, h in enumerate(helices):
            output += structure.capitalize() + " {} has: ".format(i)
            output += "{} based pairs, ".format(h["num_pairs"])
            #output_dict["helix_{}".format(i)]["base_pairs"] = h["num_pairs"]
            output_dict.setdefault(f"helix_{i}", {})["base_pairs"] = h["num_pairs"]
            output += "{} is its first strand, ".format(h["strand1"])
            output_dict.setdefault(f"helix_{i}", {})["strand_1"] = h["strand1"]
            output += "{} is its second strand, ".format(h["strand2"])
            output_dict.setdefault(f"helix_{i}", {})["strand_2"] = h["strand2"]
            output += "{} is its helix form. ".format(h['helix_form'], structure)
            output_dict.setdefault(f"helix_{i}", {})['helix_form'] = h['helix_form']

        #prompt = (
        #    "Here is minimal helix data from an RNA structure:\n"
        #    f"{json.dumps(minimal, indent=2)}\n\n"
        #    "to which type of ncRNA it corresponds?"

        #)
        return output_dict
        # return output
    
    def get_helicesInfo(self):
        if "helices" in self.json_data.keys():
            return self.summarize_stems_minimal(self.json_data["helices"], structure="helix")
        else:
            return ""
    
    def get_stemsInfo(self):
        return self.summarize_stems_minimal(self.json_data["stems"], structure="stem")
    

    def get_genInfo(self):
        #print(len(self.json_data.keys()), self.json_data.keys())
        output = ""
        output_dict = {}
        ## output_dict["full_RNA_sequence"] = self.sequence

        L = self.json_data['num_nts']
        output += "This RNA consists of {} nucleotides. ".format(L)
        output_dict["length"] = L

        if ('num_pairs' in self.json_data.keys()) and ('num_hbonds' in self.json_data.keys()):
            Npairs = self.json_data['num_pairs']
            NHbonds = self.json_data['num_hbonds']
            output += "These nucleotides form {} base pairs and {} hydrogen bonds. ".format(Npairs, NHbonds)
            output_dict["base_pairs"] = Npairs
            output_dict["hydrogen_bonds"] = NHbonds

        if 'chains' in self.json_data.keys():
            DBN = [self.json_data['chains'][chain]['sstr'] for chain in self.json_data['chains'].keys()]
            output += "The dot-bracket notation (also known as DBN) representing its secondary structure is: {}. ".format(DBN)
            output_dict["dot_bracket"] = DBN
        
        #A multiplet is a group of three or more nucleotides that:
        #Interact through hydrogen bonding or planar stacking,
        #Often form tertiary structure motifs,
        #Can be base triples, quadruples, etc.
        if "num_multiplets" in self.json_data.keys():
            output += "This RNA has {} multiplets. ".format(self.json_data['num_multiplets'])
            # output_dict["num_multiplets"] = self.json_data['num_multiplets']
            output += "More precisely: {}. ".format([el['nts_short'] for el in self.json_data['multiplets']])
            output_dict['multiplets'] = [el['nts_short'] for el in self.json_data['multiplets']]


        # specific distortions in base stacking geometry—typically places where the normal stacking between 
        # nucleotides is disrupted or widened, like a "splayed open" structure.
        if "num_splayUnits" in self.json_data.keys():
            output += "This RNA has {} splayUnits. ".format(self.json_data['num_splayUnits'])
            output += "More precisely: {}. ".format([el['nts_short'] for el in self.json_data['splayUnits']])
            output_dict['splayUnits'] = [el['nts_short'] for el in self.json_data['splayUnits']]

        #Ribose zippers are tertiary interactions in RNA where the ribose sugars of two RNA strands form hydrogen bonds, 
        #effectively “zipping” the strands together.
        #They occur when the 2′-OH groups and sugar edges of adjacent riboses from different RNA segments interact, 
        #stabilizing the 3D fold.
        #These interactions connect two RNA helices or loops, often helping fold complex RNA architectures.
        if 'num_riboseZippers' in self.json_data.keys():
            output += "This RNA has {} riboseZippers. ".format(self.json_data['num_riboseZippers'])
            output += "More precisely: {}. ".format([el['nts_short'] for el in self.json_data['riboseZippers']])
            output_dict['riboseZippers'] = [el['nts_short'] for el in self.json_data['riboseZippers']]

        #Types of single-stranded segments
        #Hairpin loops: Unpaired nucleotides closing a stem.
        #Internal loops: Unpaired nucleotides on both strands between two stems.
        #Bulges: Unpaired nucleotides on one strand only.
        #Junction loops: Single-stranded regions at multi-helix junctions.
        #Single-stranded tails: Unpaired ends of RNA strands.
        if 'num_ssSegments' in self.json_data.keys():
            output += "This RNA has {} ssSegments. ".format(self.json_data['num_ssSegments'])
            output += "More precisely: {}. ".format([el['nts_short'] for el in self.json_data['ssSegments']])
            output_dict['ssSegments'] = [el['nts_short'] for el in self.json_data['ssSegments']]

        # Junctions are regions where three or more RNA helices (stems) converge.
        if 'num_junctions' in self.json_data.keys():
            output += "This RNA has {} junctions. ".format(self.json_data['num_junctions'])
            output += "More precisely: {}. ".format([el['nts_short'] for el in self.json_data['junctions']])
            output_dict['junctions'] = [el['nts_short'] for el in self.json_data['junctions']]

        if 'num_hairpins' in self.json_data.keys():
            output += "This RNA has {} hairpins. ".format(self.json_data['num_hairpins'])
            output += "More precisely: {}. ".format([el['nts_short'] for el in self.json_data['hairpins']])
            output_dict['hairpins'] = [el['nts_short'] for el in self.json_data['hairpins']]





        if 'nonStack' in self.json_data.keys():
            N = self.json_data['nonStack']['num_nts']
            nts = self.json_data['nonStack']['nts_short']
            output += "{} nucleotides, not stacked with any neighboring base, are: {}. ".format(N, nts)
            output_dict['nonStack'] = [nts]

        if 'num_stacks' in self.json_data.keys():
            output += "This RNA has {} stacks. ".format(self.json_data['num_stacks'])
            output += "More precisely: {}. ".format([el['nts_short'] for el in self.json_data['stacks']])
            output_dict['stacks'] = [el['nts_short'] for el in self.json_data['stacks']]


        # dict_keys(['index', 'helix_index', 'num_stems', 'stem_indices'])
        # coaxial stacking interactions between helices or stems.
        if 'num_coaxStacks' in self.json_data.keys():
            output += "This RNA has {} coaxStacks. ".format(self.json_data['num_coaxStacks'])
            output_dict['num_coaxStacks'] = self.json_data['num_coaxStacks']



        #return(output)
        return(output_dict)
        #return(self.json_data)
