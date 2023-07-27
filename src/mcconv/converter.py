
import networkx as nx
import mdtraj
from .consts import basis_Zs


def residue_gas_converter(top: mdtraj.Topology, mask=None):
    G = nx.DiGraph()
    
    # first find root nodes, which are N, CA, C atoms
    backbone_atom_names = ['CA', 'C', 'N']
    backbone_atom_ids = top.select(' '.join(["name " + atom_name for atom_name in backbone_atom_names]))
    G.add_nodes_from([(i, {'ic': i not in backbone_atom_ids}) for i in range(top.n_atoms)])
    
    # implement atom mask later
    # implement multi-chain processing later
    for i, res in enumerate(top.residues):
        atom_name_to_id = {a.name: a.index for a in res.atoms}
        for entry in basis_Zs[res.name]:  # template entry:
            try:
                cur_id = atom_name_to_id[entry[0]]
                G.add_edges_from([(atom_name_to_id[refatom], cur_id, {'id': 2-j}) for j, refatom in enumerate(entry[1:])]) # j = 0,1,2
            except:
                continue
            
        
        # implement any fix later
        if i == 0: # c-terminal
            G.add_edges_from([(atom_name_to_id['N'], atom_name_to_id['H2'], {'id': 2}),
                              (atom_name_to_id['CA'], atom_name_to_id['H2'], {'id': 1}),
                              (atom_name_to_id['C'], atom_name_to_id['H2'], {'id': 0})])
            G.add_edges_from([(atom_name_to_id['N'], atom_name_to_id['H3'], {'id': 2}),
                              (atom_name_to_id['CA'], atom_name_to_id['H3'], {'id': 1}),
                              (atom_name_to_id['C'], atom_name_to_id['H3'], {'id': 0})])
        elif i == top.n_residues-1: # n-term
            G.add_edges_from([(atom_name_to_id['C'], atom_name_to_id['OXT'], {'id': 2}),
                              (atom_name_to_id['CA'], atom_name_to_id['OXT'], {'id': 1}),
                              (atom_name_to_id['N'], atom_name_to_id['OXT'], {'id': 0})])
            
    return G

def sequential_converter(top: mdtraj.Topology) -> nx.DiGraph:
    G = nx.DiGraph()

    # first find root nodes, which are N, CA, C atoms
    backbone_atom_names = ['CA', 'C', 'N']
    backbone_atom_ids = top.select(' '.join(["name " + atom_name for atom_name in backbone_atom_names]))
    backbone_atom_ids.sort()
        
    # we only use cartesian coordinate on the first three backbone atoms
    G.add_nodes_from([(i, {'ic': i not in backbone_atom_ids[:3]}) for i in range(top.n_atoms)])
    
    # add backbone computing dependencies
    for i in range(3, len(backbone_atom_ids)):
         G.add_edges_from([(backbone_atom_ids[i-3], backbone_atom_ids[i], {'id': 0}),
                            (backbone_atom_ids[i-2], backbone_atom_ids[i], {'id': 1}),
                            (backbone_atom_ids[i-1], backbone_atom_ids[i], {'id': 2})])
    
    for i, res in enumerate(top.residues):
        atom_name_to_id = {a.name: a.index for a in res.atoms}
        
        for entry in basis_Zs[res.name]:  # template entry:
            try:
                cur_id = atom_name_to_id[entry[0]]
                G.add_edges_from([(atom_name_to_id[refatom], cur_id, {'id': 2-j}) for j, refatom in enumerate(entry[1:])]) # j = 0,1,2
            except:
                continue
        
        # implement any fix later
        if i == 0: # c-terminal
            G.add_edges_from([(atom_name_to_id['N'], atom_name_to_id['H2'], {'id': 2}),
                              (atom_name_to_id['CA'], atom_name_to_id['H2'], {'id': 1}),
                              (atom_name_to_id['C'], atom_name_to_id['H2'], {'id': 0})])
            G.add_edges_from([(atom_name_to_id['N'], atom_name_to_id['H3'], {'id': 2}),
                              (atom_name_to_id['CA'], atom_name_to_id['H3'], {'id': 1}),
                              (atom_name_to_id['C'], atom_name_to_id['H3'], {'id': 0})])
        elif i == top.n_residues-1: # n-term
            G.add_edges_from([(atom_name_to_id['C'], atom_name_to_id['OXT'], {'id': 2}),
                              (atom_name_to_id['CA'], atom_name_to_id['OXT'], {'id': 1}),
                              (atom_name_to_id['N'], atom_name_to_id['OXT'], {'id': 0})])
        
    
    return G

