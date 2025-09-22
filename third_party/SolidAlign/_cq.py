# Author: Doris A. et al, https://github.com/anniedoris/CAD-Coder

import cadquery as cq
import numpy as np
from typing import Tuple, Union

def align_shapes(source : cq.Workplane, target : cq.Workplane) -> Tuple[cq.Workplane, float]:
    """Align source to target using the center of mass and the principal axes of inertia. also return normalized IOU"""
    c_source = cq.Shape.centerOfMass(source.val())
    c_target = cq.Shape.centerOfMass(target.val())

    I_source = np.array(cq.Shape.matrixOfInertia(source.val()))
    I_target = np.array(cq.Shape.matrixOfInertia(target.val()))

    v_source = cq.Shape.computeMass(source.val())
    v_target = cq.Shape.computeMass(target.val())

    I_p_source, I_v_source = np.linalg.eigh(I_source)
    I_p_target, I_v_target = np.linalg.eigh(I_target)

    s_source = np.sqrt(np.abs(I_p_source).sum()/v_source)
    s_target = np.sqrt(np.abs(I_p_target).sum()/v_target)

    if np.isnan(s_source) or np.isnan(s_target):
        raise Exception("Error eigenvalue stuff")

    normalized_source = source.translate(-c_source).val().scale(1/s_source)
    normalized_target = target.translate(-c_target).val().scale(1/s_target)

    Rs = np.zeros((4,3,3))
    Rs[0] = I_v_target @ I_v_source.T

    for i in range(3):
        # all possible 2 out of 3 permutations
        alignment = 1 - 2 * np.array([i>0, (i+1)%2, i%3<=1])
        Rs[i+1] = I_v_target @ (alignment[None,:] * I_v_source).T

    best_IOU = 0
    best_T = None
    for i in range(4):
        T = np.zeros([4,4])
        T[:3,:3] = Rs[i]
        T[-1,-1] = 1
        
        aligned_source = normalized_source.transformGeometry(cq.Matrix(T.tolist()))
        
        intersect = aligned_source.intersect(normalized_target)
        union = aligned_source.fuse(normalized_target)

        # if union.Volume() <= 0:
        #     print()
        if intersect.Volume() <= 0 or union.Volume() <= 0:
            IOU = 0
        else:
            IOU = intersect.Volume() / union.Volume()
        
        if IOU > best_IOU:
            best_IOU = IOU
            best_T = T

    if best_T is None:
        return None, 0.0

    aligned_source = normalized_source.transformGeometry(cq.Matrix(best_T.tolist())).scale(s_target).translate(c_target)
    
    return cq.Workplane(aligned_source), best_IOU