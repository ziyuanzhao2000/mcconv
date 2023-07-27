import networkx as nx
import mdtraj
from typing import Callable, List
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
from .coordsys import CoordinateSystem, AffineFrame, ref_spherical_frame
from .utils import concatenate_trees

def concatenate_frames(frames: List[CoordinateSystem]) -> CoordinateSystem:
    # assert check_same_type(frames)
    return concatenate_trees(frames, axis=1)

def sort_frame(frame: CoordinateSystem, perm) -> CoordinateSystem:
    leaves, treedef = tree_flatten(frame)
    leaves_sorted = []
    for leave in leaves:
        leaves_sorted.append(leave[:, perm, ...])
    return treedef.unflatten(leaves_sorted)

def DAG_to_Z_indices(DAG: nx.DiGraph, n_atoms: int):
    """
    returns Z_indices: n_atoms * 4 array of masked Z index (zero for Cartesian repr-ed atom rows)
    """
    Z_indices = np.zeros((n_atoms, 4), dtype=np.int32)
    for (ref_atom_id, cur_atom_id, d) in DAG.edges(data=True):
        Z_indices[cur_atom_id, 1+d['id']] = ref_atom_id
        Z_indices[cur_atom_id, 0] = cur_atom_id
    return Z_indices

def DAG_to_Z_index_groups(DAG: nx.DiGraph):
    groups = []
    for layer, nodes in enumerate(nx.topological_generations(DAG)):
        n_atoms = len(nodes)
        Z_indices = np.zeros((n_atoms, 4), dtype=np.int32)
        for row_id, cur_atom_id in enumerate(nodes):
            if DAG.nodes[cur_atom_id]['layer'] == 0:
                Z_indices[row_id, 0] = cur_atom_id
                Z_indices[row_id, 1:] = -1
            else:
                for (ref_atom_id, _, d) in DAG.in_edges(cur_atom_id, data=True):
                    Z_indices[row_id, 1+d['id']] = ref_atom_id
                    Z_indices[row_id, 0] = cur_atom_id
        groups.append(Z_indices)
    return groups

class ICTransformation():
    
    def __init__(self, top: mdtraj.Topology, DAG: nx.DiGraph=None, 
                 conversion_scheme: Callable[[mdtraj.Topology], nx.DiGraph]=None,
                 build_frame: Callable[[jnp.array, jnp.array, jnp.array], CoordinateSystem]=None):
        self.topology = top
        self.n_atoms = top.n_atoms
        self.DAG = DAG if DAG is not None else conversion_scheme(self.topology)
        self.Z_index_groups = DAG_to_Z_index_groups(self.DAG)
        self.flat_Z_indices = jnp.concatenate([Z_indices[..., 0] for Z_indices in self.Z_index_groups])
        self.ic_indices = jnp.array([i for i,n in self.DAG.nodes(data=True) if n['ic']==True], dtype=np.int32)
        self.n_groups = len(self.Z_index_groups)
        self.Z_index_group_sizes = [len(group) for group in self.Z_index_groups]
        self.build_frame = build_frame
    
    def get_frame_and_xyz_local(self, xyz):
        frame_lst = []
        xyz_transformed_lst = []

        for layer in range(self.n_groups):
            Z_indices = self.Z_index_groups[layer]
            positions = xyz[..., Z_indices[:, 0], :]
            if layer == 0:
                R0 = jnp.tile(jnp.eye(3)[None, None,...], (positions.shape[0], positions.shape[1], 1, 1))
                # assumes first layer start from the default global frame (Cartesian)
                frame = AffineFrame(R=R0, Ri=R0,
                                    t=jnp.zeros((positions.shape[0], positions.shape[1], 3)))
                xyz_transformed = positions
            else:
                frame = self.build_frame(xyz[..., Z_indices[:, 1], :], 
                                         xyz[..., Z_indices[:, 2], :], 
                                         xyz[..., Z_indices[:, 3], :]) # ref positions
            xyz_transformed = frame.to_local(positions)
            frame_lst.append(frame)
            xyz_transformed_lst.append(xyz_transformed)

        # stitch results from each layer into output
        frames = concatenate_frames(frame_lst)
        xyz_transformed = jnp.concatenate(xyz_transformed_lst, axis=1)

        perm = jnp.argsort(self.flat_Z_indices)
        frames = sort_frame(frames, perm)
        xyz_transformed = xyz_transformed[..., perm, :]
        return frames, xyz_transformed
        
    def get_frame_and_xyz_global(self, xyz):
        frame_lst = []

        for layer in range(self.n_groups):
            Z_indices = self.Z_index_groups[layer]
            positions = xyz[..., Z_indices[:, 0], :]
            if layer == 0:
                R0 = jnp.tile(jnp.eye(3)[None, None,...], (positions.shape[0], positions.shape[1], 1, 1))
                # assumes first layer start from the default global frame (Cartesian)
                frame = AffineFrame(R=R0, Ri=R0,
                                    t=jnp.zeros((positions.shape[0], positions.shape[1], 3)))
                xyz_transformed = positions
            else:
                frame = self.build_frame(xyz[..., Z_indices[:, 1], :], 
                                         xyz[..., Z_indices[:, 2], :], 
                                         xyz[..., Z_indices[:, 3], :]) # ref positions
                xyz_transformed = frame.to_global(positions)
            frame_lst.append(frame)
            xyz = xyz.at[..., Z_indices[:, 0], :].set(xyz_transformed)

        # stitch results from each layer into output
        frames = concatenate_frames(frame_lst)
        perm = jnp.argsort(self.flat_Z_indices)
        frames = sort_frame(frames, perm)
        return frames, xyz
    
    def xyz2ic(self, xyz: jnp.array, vec_angles=False):
        """
        xyz : n_batch * n_atoms * 3 array of Cartesian coordinates
        """
        
        frames, xyz_local = self.get_frame_and_xyz_local(xyz)
        
        # convert some to IC coordinates
        ic = xyz_local.at[..., self.ic_indices, :].set(\
            ref_spherical_frame.to_local(xyz_local[..., self.ic_indices, :])
        )
                
        if vec_angles:
            pass
        
        return frames, ic
            
    # need to deal with pure IC repr later
    def ic2xyz(self, ic):
        """
        ic: n_batch * n_atoms * 3 array of mixed coordinates 
        (Cartesian coordinates are used to initialize computation)
        """
        xyz_local = ic.at[..., self.ic_indices, :].set(\
            ref_spherical_frame.to_global(ic[..., self.ic_indices, :])
        )
        
        frames, xyz = self.get_frame_and_xyz_global(xyz_local)
        return frames, xyz