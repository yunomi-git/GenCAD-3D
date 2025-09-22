from cadlib.util import load_cad_vec
import cadlib.curves
from GenCADGenerator import generate_cad
from cadlib import extrude
from .onshapeInterface import ProcessUrl as RequestUrlCreator
from GenCADGenerator.generate_cad import SketchEntityLine, SketchFeature, FeatureExtrude, get_microversion, SketchEntityCircle, SketchEntityArc
from .onshapeInterface import PartStudios
import argparse

parser = argparse.ArgumentParser(description="Generate a cad program in a desired onshape part studio")
parser.add_argument("-url", "--part_studio_url", type=str, required=True, help="url of the destination part studio")
parser.add_argument("-cad_path", "--cad_path", type=str, required=True, help="path to the cad program to translate")
args = parser.parse_args()

if __name__ == "__main__":
    url = args.part_studio_url
    url = RequestUrlCreator.OnshapeUrl(url)
    cad_program_path = args.cad_path # path/to/cad_program.h5
    print(cad_program_path)
    combined = load_cad_vec(data_path="",
                            data_id=cad_program_path[:cad_program_path.rfind(".")],
                            as_single_vec=True, as_tensor=False)

    sequence = extrude.CADSequence.from_vector(combined, is_numerical=True)

    num = 1
    for extrude_seq in sequence.seq:
        print("Extrude", num)
        example_profile = extrude_seq.profile
        example_profile.denormalize(extrude_seq.sketch_size)

        # This grabs the sketch plane
        sketch_origin = extrude_seq.sketch_plane.origin
        sketch_position = extrude_seq.sketch_pos
        theta = extrude_seq.sketch_plane._theta
        phi = extrude_seq.sketch_plane._phi
        gamma = extrude_seq.sketch_plane._gamma

        sketch_plane, microversion = generate_cad.create_sketch_plane(str(num), url, z1=phi, x1=theta, z2=gamma, px=sketch_position[0], py=sketch_position[1], pz=sketch_position[2])

        # Now do the sketch
        sketch = SketchFeature("Sketch" + str(num), plane_geometry_id=sketch_plane)

        sketch_idx = 0
        for loop in example_profile.children:
            last_entity_name = None
            translation = loop.global_trans
            for sketch_entity in loop.children:
                entity_name = "sketch_entity_" + str(sketch_idx)
                if isinstance(sketch_entity, cadlib.curves.Line):
                    line = SketchEntityLine(start=sketch_entity.start_point, stop=sketch_entity.end_point, name=entity_name)
                    sketch.add_entity(line)

                elif isinstance(sketch_entity, cadlib.curves.Circle):
                    circle = SketchEntityCircle(name=entity_name, center=sketch_entity.center, radius=sketch_entity.radius)
                    sketch.add_entity(circle)

                else:
                    arc = SketchEntityArc(name=entity_name, center=sketch_entity.center,
                                                end_angle=sketch_entity.end_angle, radius=sketch_entity.radius,
                                                start_angle=sketch_entity.start_angle, clock_sign=not bool(sketch_entity.clock_sign),
                                                ref_vec=sketch_entity.ref_vec)
                    sketch.add_entity(arc)

                # if last_entity_name is not None:
                #     sketch.add_constraint(SketchConstraintCoincident(last_entity_name, entity_name))
                last_entity_name = entity_name
                sketch_idx += 1
        feature = sketch.get_json()
        addFeature = PartStudios.AddFeature(url)
        addFeature.json_feature = feature
        addFeature.source_microversion = microversion
        addFeature.send_request()

        # Sketch created. Now extrude
        microversion = get_microversion(url)

        ev_featurescript = PartStudios.EvaluateFeaturescipt(url)
        ev_featurescript.source_microversion = microversion
        ev_featurescript.set_query_sketch_faces()
        output = ev_featurescript.send_request()
        query = ev_featurescript.get_query_result(output)

        if extrude_seq.extent_one < 0:
            extrude_seq.extent_one *= -1 # TODO verify this
        extrude = FeatureExtrude("Extrude " + str(num), query[-1], extrude_seq.extent_one, extrude_seq.extent_two, extrude_seq.operation, extrude_seq.extent_type)
        feature = extrude.get_json()
        addFeature = PartStudios.AddFeature(url)
        addFeature.json_feature = feature
        addFeature.source_microversion = microversion
        addFeature.send_request()


        num += 1
    print()