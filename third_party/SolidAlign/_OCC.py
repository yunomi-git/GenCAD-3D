# Author: Doris A. et al, https://github.com/anniedoris/CAD-Coder

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.gp import gp_Vec, gp_Pnt, gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCC.Core.TopoDS import TopoDS_Shape
import numpy as np
from typing import Tuple

def load_step_file(filename : str) -> TopoDS_Shape:
    """Load a STEP file and return the shape."""
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)
    if status != IFSelect_RetDone:
        raise Exception("Error: Cannot read STEP file." + filename)
    # Transfer the roots and get the shape
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    return shape

def compute_mass_properties(shape : TopoDS_Shape) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute mass properties such as volume (interpreted as mass for unit density)
    and the center of mass of the given shape."""
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    mass = props.Mass()  # For solids, this gives the volume (mass = volume * density)
    center_of_mass = props.CentreOfMass()
    center_of_mass = np.array([center_of_mass.X(), center_of_mass.Y(), center_of_mass.Z()])
    matrix_of_inertia = props.MatrixOfInertia()
    matrix_of_inertia = np.array([[matrix_of_inertia.Value(1, 1), matrix_of_inertia.Value(1, 2), matrix_of_inertia.Value(1, 3)],
                                  [matrix_of_inertia.Value(2, 1), matrix_of_inertia.Value(2, 2), matrix_of_inertia.Value(2, 3)],
                                  [matrix_of_inertia.Value(3, 1), matrix_of_inertia.Value(3, 2), matrix_of_inertia.Value(3, 3)]])
    return mass, center_of_mass, matrix_of_inertia

def align_shapes(source : TopoDS_Shape, target : TopoDS_Shape) -> Tuple[TopoDS_Shape, float]:
    """Align source to target using the center of mass and the principal axes of inertia. also return normalized IOU"""
    
    m1, c1, mat1 = compute_mass_properties(target)
    m2, c2, mat2 = compute_mass_properties(source)

    eig1, v1 = np.linalg.eigh(mat1)
    eig2, v2 = np.linalg.eigh(mat2)

    if m1 <= 0 or m2 <= 0:
        raise Exception("m1 m2 < 0")
    s1 = np.sqrt(np.abs(eig1).sum()/m1)
    s2 = np.sqrt(np.abs(eig2).sum()/m2)

    translation_vector = gp_Vec(-c1[0], -c1[1], -c1[2])
    translation = gp_Trsf()
    translation.SetTranslation(translation_vector)
    shape_1 = BRepBuilderAPI_Transform(target, translation, True).Shape()

    scaling = gp_Trsf()
    scaling.SetScale(gp_Pnt(0,0,0), 1/s1)
    shape_1 = BRepBuilderAPI_Transform(shape_1, scaling, True).Shape()

    translation_vector = gp_Vec(-c2[0], -c2[1], -c2[2])
    translation = gp_Trsf()
    translation.SetTranslation(translation_vector)
    shape_2 = BRepBuilderAPI_Transform(source, translation, True).Shape()
    scaling = gp_Trsf()
    scaling.SetScale(gp_Pnt(0,0,0), 1/s2)
    shape_2 = BRepBuilderAPI_Transform(shape_2, scaling, True).Shape()

    Rs = np.zeros((4,3,3))
    Rs[0] = v1 @ v2.T

    for i in range(3):
        # all possible 2 out of 3 permutations
        alignment = 1 - 2 * np.array([i>0, (i+1)%2, i%3<=1])
        Rs[i+1] = v1 @ (alignment[None,:] * v2).T
    
    
    best_IOU = 0
    best_T = None
    for i in range(4):
        R = Rs[i]
        rotation = gp_Trsf()
        rotation.SetValues(R[0,0], R[0,1], R[0,2], 0,
                            R[1,0], R[1,1], R[1,2], 0,
                            R[2,0], R[2,1], R[2,2], 0)
        
        shape_2_aligned = BRepBuilderAPI_Transform(shape_2, rotation, True).Shape()

        intersection = BRepAlgoAPI_Common(shape_1, shape_2_aligned).Shape()
        union = BRepAlgoAPI_Fuse(shape_1, shape_2_aligned).Shape()

        V_I, _, _ = compute_mass_properties(intersection)
        V_U, _, _ = compute_mass_properties(union)
        if V_U == 0:
            print("0 Union calculated ")
            continue

        IOU = V_I/V_U
        
        if IOU >= best_IOU:
            best_IOU = IOU
            best_T = R
        
    # rotation to align 2 on 1
    if best_IOU == 0:
        print("0 IOU calculated")
    R = best_T
    rotation = gp_Trsf()
    rotation.SetValues(R[0,0], R[0,1], R[0,2], 0,
                        R[1,0], R[1,1], R[1,2], 0,
                        R[2,0], R[2,1], R[2,2], 0)
    shape_2 = BRepBuilderAPI_Transform(shape_2, rotation, True).Shape()
    
    # rescale and translate to target frame
    scaling = gp_Trsf()
    scaling.SetScale(gp_Pnt(0,0,0), s1)
    shape_2 = BRepBuilderAPI_Transform(shape_2, scaling, True).Shape()
    translation_vector = gp_Vec(c1[0], c1[1], c1[2])
    translation = gp_Trsf()
    translation.SetTranslation(translation_vector)
    shape_2 = BRepBuilderAPI_Transform(shape_2, translation, True).Shape()
    
    return shape_2, best_IOU
 