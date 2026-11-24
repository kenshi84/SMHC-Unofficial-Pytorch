import json
import os
import shutil
import sys

import bpy
import numpy as np
from mathutils import Matrix

# inputs to the script are supposed to be:
#   1- path to the head PLY in millimeters, in the same Y-up frame as the CT2Hair .bin
#   2- path to save the rendering results
#   3- images resolution height
#   4- images resoltuion width
#   5- strand bevel radius in meter
#   6- samples number in ray tracing
#   7- constant number of points per strands, with default -1, which necessitates presence of per strand point list
#   8- path to the site-packages folder in the smhc_torch python enivironment

argv = sys.argv
argv = argv[argv.index("--args") + 1 :]

head_model_path = argv[0]
output_folder_ds = argv[1]
strands_folder = argv[1]
temp_folder = os.path.join(output_folder_ds, "temp")
image_height = int(argv[2])
image_width = int(argv[3])
strands_thickness = float(argv[4])
samles_number = int(argv[5])
constant_points_num_per_strand = int(argv[6])

sys.path.append(argv[7])

import trimesh


# Right-handed change of basis: (x, y, z) -> (x, -z, y).
Y_UP_TO_Z_UP = np.array(((1, 0, 0), (0, 0, -1), (0, 1, 0)), dtype=np.float32)


def create_hair(
    name: str,
    strands_points: np.ndarray,
    points_per_strand: np.ndarray,
    fixed_points_num_per_strand=-1,
):

    def create_points(curveData, coords, index=0):
        polyline = curveData.splines.new("POLY")
        polyline.points.add(len(coords) - 1)
        for i, coord in enumerate(coords):
            x, y, z = coord
            polyline.points[index].co = (x, y, z, i)
            index += 1

    curveData = bpy.data.curves.new("hair", type="CURVE")
    curveData.dimensions = "3D"
    curveData.resolution_u = 1

    total_strands = 0
    if fixed_points_num_per_strand > 0:
        strands_arr = strands_points.reshape(-1, fixed_points_num_per_strand, 3)
        for i in range(strands_arr.shape[0]):
            strand = strands_arr[i].copy()
            total_strands += 1
            create_points(curveData, strand)
    elif points_per_strand is not None:
        ps_acc = 0
        for idx, ps in enumerate(points_per_strand):
            strand = strands_points[ps_acc : ps_acc + ps, :].copy()
            ps_acc += ps
            total_strands += 1
            create_points(curveData, strand)
    else:
        raise ValueError(
            "To extract strands from the provided point cloud either number of points in each strand must be provided or strands must have fixed number of points per each one of them!"
        )

    print(f"total number of strands is {total_strands}")

    curveOB = bpy.data.objects.new(name, curveData)
    bpy.data.scenes[0].collection.objects.link(curveOB)

    # curve_object.data.extrude = 0.0002
    curveOB.data.bevel_mode = (
        "ROUND"  # Gives it a circle shape without needing another object as a profile.
    )
    curveOB.data.bevel_depth = 0.0002  # 1 Meter Radius (NOT DIAMETER)
    curveOB.data.bevel_resolution = 0

    # reset state of selection
    for obj in bpy.context.selected_objects:
        obj.select_set(False)

    bpy.context.view_layer.objects.active = curveOB
    curveOB.select_set(True)
    bpy.ops.object.convert(target="MESH")

    return bpy.data.objects[name]


def create_folder(folder):
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as error:
        pass


if __name__ == "__main__":

    # enable usage of gpus
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = list(cycles_preferences.devices)[:2]

    activated_gpus = []
    for device in devices:
        device.use = True
        activated_gpus.append(device.name)

    cycles_preferences.compute_device_type = "OPTIX"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.render.use_persistent_data = True

    # camera views are set based on the vertices of the icosphere
    ico_sphere = bpy.data.objects["icosphere"]
    ico_vertices = ico_sphere.data.vertices

    # creating hair strands
    print("Creating hair strands ...")
    strands_file = os.path.join(strands_folder, "strands_interpolated.ply")
    points_per_strand_file = os.path.join(
        strands_folder, "strands_interpolated_points_per_strand.npy"
    )

    if (
        not os.path.exists(points_per_strand_file)
        and constant_points_num_per_strand < 0
    ):
        raise ValueError(
            "To extract strands from the provided point cloud either number of points in each strand must be provided or strands must have fixed number of points per each one of them!"
        )

    with open(strands_file, "rb") as f:
        strands_mesh = trimesh.exchange.ply.load_ply(f)
    # load_hair_models.py has already converted the .bin strand positions to meters.
    strands = strands_mesh["vertices"] @ Y_UP_TO_Z_UP.T

    if os.path.exists(points_per_strand_file):
        with open(points_per_strand_file, "rb") as f:
            points_per_strand = np.load(f).astype(int)
        hair_objects = create_hair(
            "Hair", strands, points_per_strand, fixed_points_num_per_strand=-1
        )
    else:
        hair_objects = create_hair(
            "Hair",
            strands,
            points_per_strand=None,
            fixed_points_num_per_strand=constant_points_num_per_strand,
        )

    # Assign it to object
    hair_mat = bpy.data.materials.get("hair_material")
    # hair_mat = bpy.data.materials.get("strands_material")

    if hair_objects.data.materials:
        # assign to 1st material slot
        hair_objects.data.materials[0] = hair_mat
    else:
        # no slots
        hair_objects.data.materials.append(hair_mat)

    print("Loading the head model ...")
    bpy.ops.wm.ply_import(filepath=head_model_path)
    head_object = bpy.context.active_object
    head_object.data.transform(Matrix(0.001 * Y_UP_TO_Z_UP).to_4x4())
    head_mat = bpy.data.materials.get("head_material")
    # head_mat = bpy.data.materials.get("Fresnel_Material")

    if head_object.data.materials:
        # assign to 1st material slot
        head_object.data.materials[0] = head_mat
    else:
        # no slots
        head_object.data.materials.append(head_mat)

    # Scene config
    center = (0.0, 0.0, 0.0)
    radius = 0.9

    # Lights config
    power = 1.5
    light_sphere_radius = radius

    bpy.data.materials["emission_material"].node_tree.nodes["Emission"].inputs[
        1
    ].default_value = power

    bpy.data.objects["Light Sphere"].location = center
    bpy.data.objects["Light Sphere"].scale = (light_sphere_radius,) * 3

    # Rendering config
    bpy.data.objects["Target"].location = center
    resolution = (int(image_width), int(image_height))
    samples = int(samles_number)
    z_near = 0.1
    z_far = 100.0

    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = (
        resolution
    )
    bpy.context.scene.cycles.samples = samples
    bpy.data.objects["Camera"].data.clip_start = z_near
    bpy.data.objects["Camera"].data.clip_end = z_far

    create_folder(temp_folder)

    print("Rendering ...")
    for i in range(len(ico_vertices)):
        print(f"rendering view {i}")

        bpy.data.objects["Camera"].location = ico_vertices[i].co * radius

        bpy.data.scenes["Scene"].node_tree.nodes["File Output"].base_path = (
            os.path.join(temp_folder, str(i))
        )
        bpy.data.scenes["Scene"].node_tree.nodes["File Output Depth"].base_path = (
            os.path.join(temp_folder, str(i))
        )
        bpy.ops.render.render(write_still=True)

    print("rendering complete")
    create_folder(output_folder_ds)
    create_folder(os.path.join(output_folder_ds, "image"))
    create_folder(os.path.join(output_folder_ds, "hair"))
    create_folder(os.path.join(output_folder_ds, "depth"))

    for i in range(len(ico_vertices)):
        print(f"copying data from the temp folders for view {i}")

        shutil.copyfile(
            os.path.join(temp_folder, str(i), "image0001.png"),
            os.path.join(output_folder_ds, "image", f"img_{i:06d}.png"),
        )

        shutil.copyfile(
            os.path.join(temp_folder, str(i), "hair0001.png"),
            os.path.join(output_folder_ds, "hair", f"img_{i:06d}.png"),
        )

        exr_path = os.path.join(temp_folder, str(i), "depth0001.exr")
        exr_image = bpy.data.images.load(exr_path)
        pixels = np.array(bpy.data.images["depth0001.exr"].pixels)
        print(f"loaded depth image pixels have the shape {pixels.shape}")
        pixels = pixels.reshape(resolution[0], resolution[0], 4)
        with open(
            os.path.join(output_folder_ds, "depth", f"depth_{i:06d}.npy"), "wb"
        ) as f:
            # the loaded exr depth image contains depth information repeated in the first three channels and constant value of 1 for the alpha channel
            # so there is no difference picking the pixels[:, :, 0] or pixels[:, :, 1] or pixels[:, :, 2]
            np.save(f, pixels[:, :, 0])

        bpy.data.images.remove(exr_image)
        shutil.rmtree(os.path.join(temp_folder, str(i)))

    # getting the camera calibration data as a json file
    scene = bpy.context.scene
    render = scene.render

    json_save_path = os.path.join(output_folder_ds, "calibration.json")

    cam_matrices = []
    camera_calib_dict = {
        "cameras": [],
        "strands_to_world_rotation": Y_UP_TO_Z_UP.tolist(),
    }

    for i in range(len(ico_vertices)):

        entry = {"camera_id": f"C{i:06d}"}
        bpy.data.objects["Camera"].location = ico_vertices[i].co * radius

        bpy.context.view_layer.update()

        modelview_matrix = bpy.data.objects["Camera"].matrix_world.inverted()

        projection_matrix = bpy.data.objects["Camera"].calc_matrix_camera(
            bpy.data.scenes["Scene"].view_layers["ViewLayer"].depsgraph,
            x=render.resolution_x,
            y=render.resolution_y,
            scale_x=render.pixel_aspect_x,
            scale_y=render.pixel_aspect_y,
        )

        cam_matrix = projection_matrix @ modelview_matrix

        cam_matrix_np = np.array(cam_matrix)
        projection_matrix_np = np.array(projection_matrix)
        modelview_matrix_np = np.array(modelview_matrix)

        extrinsic_inv = np.linalg.inv(modelview_matrix_np)
        cam_position = extrinsic_inv[:, 3]

        # converting the opengl format to opencv here
        # for intrisics matrix
        intrinsics_cv_np = np.zeros((3, 3), dtype=np.float32)
        intrinsics_cv_np[0, 0] = image_width / 2 * projection_matrix_np[0, 0]  # fx
        intrinsics_cv_np[1, 1] = image_height / 2 * projection_matrix_np[1, 1]  # fy
        intrinsics_cv_np[0, 2] = (
            image_width / 2 * (1 - projection_matrix_np[0, 2])
        )  # cx
        intrinsics_cv_np[1, 2] = (
            image_height / 2 * (1 + projection_matrix_np[1, 2])
        )  # cy
        intrinsics_cv_np[2, 2] = 1.0
        # for extrinsics matrix
        # gl_to_cv matrix is the same as cv_to_gl
        cv_to_gl = np.zeros((4, 4), dtype=np.float32)
        cv_to_gl[0, 0] = 1.0
        cv_to_gl[1, 1] = -1.0
        cv_to_gl[2, 2] = -1.0
        cv_to_gl[3, 3] = 1.0
        extrinsics_cv_np = cv_to_gl @ modelview_matrix_np

        # entry["extrinsics"] = {"view_matrix": modelview_matrix_np.reshape(-1).tolist(), "cam_position": cam_position.tolist()}
        # entry["intrinsics"] = {"camera_matrix": projection_matrix_np.reshape(-1).tolist(), "resolution": [image_width, image_height]}
        entry["extrinsics"] = {
            "view_matrix": extrinsics_cv_np.reshape(-1).tolist(),
            "cam_position": cam_position.tolist(),
        }
        entry["intrinsics"] = {
            "camera_matrix": intrinsics_cv_np.reshape(-1).tolist(),
            "resolution": [image_width, image_height],
        }

        camera_calib_dict["cameras"].append(entry)

    with open(json_save_path, "w") as f:
        json.dump(camera_calib_dict, f, indent=4)
