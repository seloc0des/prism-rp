"""Render a Prism RP scene export (project/benchmarks/prism_scene_export.gd) with Cycles.

    blender -b --factory-startup --python tests/reference/cycles_reference.py -- \
        --scene out/scene.json --out out/reference --samples 256 --diffuse-bounces 1

Writes <out>_passes.exr, a multilayer EXR with every render pass (read the "DiffInd" pass with
tests/reference/exr.py: bounced diffuse light for white albedo, the quantity Lumina's "Denoised
Irradiance" debug view shows), and <out>_combined.png for a visual check.

Deliberately avoids the compositor: its Python API changed in Blender 5.0, while multilayer EXR
output works the same across 4.x and 5.x.
"""

import argparse
import json
import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import godot_to_blender as g2b  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--diffuse-bounces", type=int, default=1)
    p.add_argument("--resolution", type=int, nargs=2, default=None, help="Override the export's resolution.")
    return p.parse_args(argv)


def make_material(name, spec, scene_dir):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    r, g, b, a = spec["albedo"]
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value = spec["roughness"]
    bsdf.inputs["Metallic"].default_value = spec["metallic"]
    e = spec["emission"]
    if max(e) > 0.0:
        bsdf.inputs["Emission Color"].default_value = (e[0], e[1], e[2], 1.0)
        bsdf.inputs["Emission Strength"].default_value = 1.0
    if spec.get("albedo_texture"):
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(os.path.join(scene_dir, spec["albedo_texture"]))
        # ShaderNodeMix (Blender 3.4+); the legacy MixRGB node is deprecated. Colour sockets are
        # inputs 6 (A) / 7 (B) and output 2 when data_type is RGBA.
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs[0].default_value = 1.0
        mix.inputs[7].default_value = (r, g, b, 1.0)
        links.new(tex.outputs["Color"], mix.inputs[6])
        links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        if spec["transparency"] == "scissor":
            cmp = nodes.new("ShaderNodeMath")
            cmp.operation = "GREATER_THAN"
            cmp.inputs[1].default_value = spec["alpha_scissor"]
            links.new(tex.outputs["Alpha"], cmp.inputs[0])
            links.new(cmp.outputs["Value"], bsdf.inputs["Alpha"])
        elif spec["transparency"] == "alpha":
            links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    elif spec["transparency"] == "alpha":
        bsdf.inputs["Alpha"].default_value = a
    return mat


def build_mesh(spec, scene_dir, index):
    verts = g2b.flat_positions(spec["positions"])
    faces = g2b.triangles_ccw(spec["indices"])
    mesh = bpy.data.meshes.new(f"m{index}_{spec['name']}")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    if spec.get("uvs"):
        uvs = spec["uvs"]
        layer = mesh.uv_layers.new(name="UVMap")
        for poly in mesh.polygons:
            for loop_index in poly.loop_indices:
                v = mesh.loops[loop_index].vertex_index
                layer.data[loop_index].uv = g2b.uv(uvs[2 * v], uvs[2 * v + 1])
    if spec["normals"]:
        normals = g2b.flat_positions(spec["normals"])
        mesh.normals_split_custom_set_from_vertices(normals)
    mesh.materials.append(make_material(f"mat{index}", spec["material"], scene_dir))
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.scene.collection.objects.link(obj)


def build_lights(lights):
    for i, l in enumerate(lights):
        if l["type"] == "omni":
            data = bpy.data.lights.new(f"omni{i}", "POINT")
            data.energy = g2b.omni_power_watts(l["energy"])
            data.shadow_soft_size = 0.0
            data.color = l["color"]
            data.use_shadow = l["shadow"]
            obj = bpy.data.objects.new(data.name, data)
            obj.location = g2b.vec(l["position"])
        elif l["type"] == "directional":
            data = bpy.data.lights.new(f"sun{i}", "SUN")
            data.energy = g2b.sun_strength(l["energy"])
            data.angle = 0.0
            data.color = l["color"]
            data.use_shadow = l["shadow"]
            obj = bpy.data.objects.new(data.name, data)
            rot = g2b.look_rotation_matrix(g2b.vec(l["direction"]))
            from mathutils import Matrix
            obj.matrix_world = Matrix(rot).to_4x4()
        else:
            continue
        bpy.context.scene.collection.objects.link(obj)


def build_world(env):
    world = bpy.data.worlds.new("PrismWorld")
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputWorld")
    # Camera rays see the background colour; everything else sees Godot's ambient colour
    # (Godot's BG_COLOR background doesn't light the scene, its ambient colour does).
    path = nodes.new("ShaderNodeLightPath")
    bg_cam = nodes.new("ShaderNodeBackground")
    bg_cam.inputs["Color"].default_value = (*env["background"], 1.0)
    bg_amb = nodes.new("ShaderNodeBackground")
    bg_amb.inputs["Color"].default_value = (*env["ambient"], 1.0)
    mix = nodes.new("ShaderNodeMixShader")
    links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    links.new(bg_amb.outputs["Background"], mix.inputs[1])
    links.new(bg_cam.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])
    bpy.context.scene.world = world


def build_camera(cam, width, height):
    from mathutils import Matrix
    data = bpy.data.cameras.new("PrismCamera")
    fit, angle = g2b.camera_fov(cam["fov_deg"], cam["keep_aspect"], width, height)
    data.sensor_fit = fit
    if fit == "VERTICAL":
        data.angle_y = angle
    else:
        data.angle_x = angle
    data.clip_start = cam["near"]
    data.clip_end = cam["far"]
    obj = bpy.data.objects.new("PrismCamera", data)
    obj.matrix_world = Matrix(g2b.camera_matrix(cam["basis_columns"], cam["origin"]))
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.camera = obj


def set_output_format(settings, multilayer_exr):
    """Blender 4.x has a distinct OPEN_EXR_MULTILAYER file format; Blender 5.0 replaced it with
    media_type = 'MULTI_LAYER_IMAGE' plus file_format = 'OPEN_EXR'."""
    if multilayer_exr:
        try:
            settings.file_format = "OPEN_EXR_MULTILAYER"
        except TypeError:
            settings.media_type = "MULTI_LAYER_IMAGE"
            settings.file_format = "OPEN_EXR"
        settings.color_depth = "32"
        settings.exr_codec = "ZIP"
    else:
        if hasattr(settings, "media_type"):
            settings.media_type = "IMAGE"
        settings.file_format = "PNG"
        settings.color_depth = "8"


def configure_render(args, width, height):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = max(args.diffuse_bounces, 1) + 2
    scene.cycles.diffuse_bounces = args.diffuse_bounces
    scene.cycles.glossy_bounces = 1
    scene.cycles.transmission_bounces = 1
    scene.cycles.transparent_max_bounces = 16
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.seed = 1
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_layers[0].use_pass_diffuse_indirect = True

    set_output_format(scene.render.image_settings, multilayer_exr=True)


def main():
    args = parse_args()
    with open(args.scene, encoding="utf-8") as f:
        data = json.load(f)
    scene_dir = os.path.dirname(os.path.abspath(args.scene))
    width, height = args.resolution or data["resolution"]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    for i, m in enumerate(data["meshes"]):
        build_mesh(m, scene_dir, i)
    build_lights(data["lights"])
    build_world(data["environment"])
    build_camera(data["camera"], width, height)
    configure_render(args, width, height)

    scene = bpy.context.scene
    scene.render.filepath = args.out + "_passes.exr"
    scene.render.use_file_extension = False
    bpy.ops.render.render(write_still=True)

    # Also save the combined image as an 8-bit PNG for a quick visual check.
    set_output_format(scene.render.image_settings, multilayer_exr=False)
    bpy.data.images["Render Result"].save_render(args.out + "_combined.png", scene=scene)
    print(f"[prism-reference] wrote {args.out}_passes.exr ({width}x{height}, Blender {bpy.app.version_string})")


main()
