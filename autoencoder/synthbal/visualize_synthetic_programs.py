import paths
from geometry.geometry_data import GeometryLoader
from cadlib.util import visualize_program
from utils.visualizer_util import DaVinciVisualizer
import utils.util as util

visualization_set = []


if __name__=="__main__":
    data_root = paths.DATA_PATH + "GenCAD3D_SynthBal/"

    geometry_loader = GeometryLoader(data_root=data_root, phase="train")
    all_data = geometry_loader.get_all_valid_data(cad=True, mesh=False, cloud=False)

    num_augmentations = 5
    num_cad = 30

    save_folder = paths.HOME_PATH + "visualization/synthbal/"
    paths.mkdir(save_folder)

    # collect examples
    raw_cad_ids = [cad_id for cad_id in all_data if cad_id.find("_") == -1]
    print(raw_cad_ids[:5])
    augmentation_dict = {}
    valid_cad_ids = []
    num_found = 0
    for cad_id in raw_cad_ids:
        augmentation_dict[cad_id] = [aug_id for aug_id in all_data if aug_id.find(cad_id[-8:] + "_") != -1]
        if len(augmentation_dict[cad_id]) >= num_augmentations:
            valid_cad_ids.append(cad_id)
            num_found += 1
        if num_found == num_cad:
            break

    visualizer = DaVinciVisualizer(geometry_loader)

    for cad_id in valid_cad_ids[:num_cad]:
        save_subfolder = save_folder
        save_3d_subfolder = save_folder + cad_id[:-8] + "/"
        visualizer.plot_picture(data_id=cad_id, style="cad", text="Input", normals=None, save_path=save_subfolder + "orig.png",
                                save_path_3d=save_3d_subfolder + "orig.gltf")
        cad_encoding = geometry_loader.load_cad(data_id=cad_id, tensor=False, as_single_vec=True)
        save_path_program = save_3d_subfolder + "orig" + ".png"
        visualize_program(cad_encoding, save_path_program, legend=True)
        pic_names = [save_subfolder + "orig.png"]

        augmentations = augmentation_dict[cad_id][:num_augmentations]
        for j in range(len(augmentations)):
            # pl.subplot(0, j + 2)
            aug_id = augmentations[j]
            cad_encoding = geometry_loader.load_cad(data_id=aug_id, tensor=False, as_single_vec=True)

            visualizer.plot_picture(data_id=aug_id, style="cad", text="Augmentation", normals=None, save_path=save_subfolder + str(j) + ".png",
                                    save_path_3d=save_3d_subfolder + str(j) + ".gltf")

            save_path_program = save_3d_subfolder + str(j) + ".png"
            visualize_program(cad_encoding, save_path_program, legend=True)

            pic_names.append(save_subfolder + str(j) + ".png")
        print("save_to", save_subfolder + cad_id[:-8] + "_comb.png")
        util.combine_images(image_paths=pic_names,
                            save_name=save_subfolder + cad_id[:-8] + "_comb.png")
        # save the vec file




