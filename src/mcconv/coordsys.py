import jax.numpy as jnp
from abc import ABC, abstractmethod
from jax.tree_util import tree_flatten

# we will abuse terminology and take frame and coordinate system to be roughly synonymous

class CoordinateSystem(ABC):
    @abstractmethod
    def to_local(self, coords):
        pass

    @abstractmethod
    def to_global(self, coords):
        pass

class AffineFrame(CoordinateSystem):
    def __init__(self, R, t, Ri=None):
        self.R = R # 3*3 
        self.Ri = jnp.linalg.inv(R) if Ri is None else Ri
        self.t = t # 3*1

    def to_local(self, coords):
        return jnp.einsum('...ik,...k->...i',
                          self.Ri, coords - self.t)

    def to_global(self, coords):
        """
        coords: n_batch * n_points * 3 
        """
        return self.t + jnp.einsum('...ik,...k->...i',
                                   self.R, coords)
    
    def _tree_flatten(self):
        children = (self.R, self.Ri, self.t)  # arrays / dynamic values
        aux_data = None
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        R, Ri, t = children
        return cls(R=R, Ri=Ri, t=t)
    
from jax import tree_util
tree_util.register_pytree_node(AffineFrame,
                               AffineFrame._tree_flatten,
                               AffineFrame._tree_unflatten)
# class CartesianFrame(AffineFrame):
  
class SphericalFrame(CoordinateSystem):
    def __init__(self, permutation=[0,1,2]):
        self.permutation = permutation
        self.inverse = tuple(permutation.index(i) for i in range(len(permutation)))
        
    
    def to_local(self, coords):
        x, y, z = coords[..., 0:1], coords[..., 1:2], coords[..., 2:3]
        r = jnp.sqrt(x*x+y*y+z*z)
        theta = jnp.arctan2(y, x)
        phi = jnp.arccos(z/r)
        out = [r, theta, phi]
        return jnp.concatenate([out[i] for i in self.permutation], axis=-1)
    
    def to_global(self, coords):
        in_ = (coords[..., 0:1], coords[..., 1:2], coords[..., 2:3])
        r, theta, phi = (in_[i] for i in self.inverse)
        x = r * jnp.cos(theta) * jnp.sin(phi)
        y = r * jnp.sin(theta) * jnp.sin(phi)
        z = r * jnp.cos(phi)
        return jnp.concatenate([x, y, z], axis=-1)

    def _tree_flatten(self):
        children = self.permutation
        aux_data = None
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        permutation = children
        return cls(permutation)

tree_util.register_pytree_node(SphericalFrame,
                               # lambda x: ((), x._tree_flatten()),
                               SphericalFrame._tree_flatten,
                               SphericalFrame._tree_unflatten)


def build_frame(x1, x2, x3) -> CoordinateSystem:
    """
    x1, x2, x3 are the three atoms before x4, whose coordinate we want to express 
    in the local frame computed from x1, x2, x3. This algorithm is based on NeRF.
    
    x1: n_atoms * 3 array
    """
    bc = x3 - x2
    bc = bc / jnp.linalg.norm(bc, axis=-1, keepdims=True) # normalize
    
    n = jnp.cross(bc, x1-x2, axisa=-1, axisb=-1)
    n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)
    
    nxbc = jnp.cross(n, bc, axisa=-1, axisb=-1)
    
    coordaxes = jnp.concatenate([nxbc[..., None], n[..., None], -bc[..., None]], axis=-1)
        
    return AffineFrame(R=coordaxes, t=x3[..., :])

    
ref_spherical_frame = SphericalFrame(permutation=(0,2,1))
