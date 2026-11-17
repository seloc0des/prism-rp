"""Render a Prism RP scene export (project/benchmarks/prism_scene_export.gd) with Cycles.

    blender -b --factory-startup --python tests/reference/cycles_reference.py -- \
        --scene out/scene.json --out out/reference --samples 256 --diffuse-bounces 1

Writes two EXRs and a PNG:
  <out>_full.exr    with --diffuse-bounces bounces
  <out>_direct.exr  with no bounces (direct light and emission only)
  <out>_full.png    for a visual check
tests/reference/exr.py:load_indirect() returns full - direct: the bounced light alone.

Every material shows white, non-metallic, non-specular to camera rays (Light Path "Is Camera
Ray") and its real look to every other ray. So the difference is bounced diffuse light for white
albedo at the visible surface, coloured by what it bounced off: the quantity Lumina's "Denoised
Irradiance" debug view shows.

Cross-check (Blender 4.0, synthetic Cornell box, 512 spp): this agrees with Cycles' own Diffuse
Indirect pass in structure (per-pixel correlation 0.96) but reads ~17-24% lower in level. The pass
separates lighting from colour by dividing out the base colour, which isn't exact for the layered
Principled BSDF; this method renders the target quantity directly. Gate 2's scale-fitted PSNR is
unaffected by a uniform level difference; the raw PSNR and fitted scale show it.

This uses only the Combined image on purpose. Render passes and the compositor's Python API both
changed in Blender 5.0; two plain renders work the same across 4.x and 5.x.
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
    bsdf.inputs["Roughness"].default_value = spec["roughness"]

    # Camera rays see a white, non-metallic, non-specular surface; other rays the real material.
    path = nodes.new("ShaderNodeLightPath")
    cam_white = nodes.new("ShaderNodeMix")
    cam_white.data_type = "RGBA"
    cam_white.blend_type = "MIX"
    cam_white.inputs[6].default_value = (r, g, b, 1.0)      # A: real albedo (a texture replaces it below)
    cam_white.inputs[7].default_value = (1.0, 1.0, 1.0, 1.0)  # B: white
    links.new(path.outputs["Is Camera Ray"], cam_white.inputs[0])
    links.new(cam_white.outputs[2], bsdf.inputs["Base Color"])
    not_cam = nodes.new("ShaderNodeMath")
    not_cam.operation = "SUBTRACT"
    not_cam.inputs[0].default_value = 1.0
    links.new(path.outputs["Is Camera Ray"], not_cam.inputs[1])
    for socket, value in (("Metallic", spec["metallic"]), ("Specular IOR Level", 0.5)):
        scaled = nodes.new("ShaderNodeMath")
        scaled.operation = "MULTIPLY"
        scaled.inputs[1].default_value = value
        links.new(not_cam.outputs["Value"], scaled.inputs[0])
        links.new(scaled.outputs["Value"], bsdf.inputs[socket])
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
        links.new(mix.outputs[2], cam_white.inputs[6])
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


def set_output_format(settings, kind):
    """kind: 'exr' (float RGBA) or 'png'. Blender 5.0 added image_settings.media_type, which limits
    the allowed file formats; set it first when it exists (4.x has none)."""
    if hasattr(settings, "media_type"):
        settings.media_type = "IMAGE"
    if kind == "exr":
        settings.file_format = "OPEN_EXR"
        settings.color_depth = "32"
        settings.exr_codec = "ZIP"
    else:
        settings.file_format = "PNG"
        settings.color_depth = "8"


def set_bounces(scene, diffuse):
    scene.cycles.diffuse_bounces = diffuse
    scene.cycles.glossy_bounces = 1 if diffuse > 0 else 0
    scene.cycles.transmission_bounces = 1 if diffuse > 0 else 0
    scene.cycles.max_bounces = diffuse + 2 if diffuse > 0 else 0
    scene.cycles.transparent_max_bounces = 16   # alpha cut-outs stay correct with no bounces


def render_to(scene, path):
    scene.render.filepath = path
    scene.render.use_file_extension = False
    bpy.ops.render.render(write_still=True)


def configure_render(args, width, height):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.seed = 1
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    set_output_format(scene.render.image_settings, "exr")


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
    set_bounces(scene, args.diffuse_bounces)
    render_to(scene, args.out + "_full.exr")
    set_output_format(scene.render.image_settings, "png")
    bpy.data.images["Render Result"].save_render(args.out + "_full.png", scene=scene)

    # Same seed, no bounces: the direct light and emission cancel in the difference.
    set_output_format(scene.render.image_settings, "exr")
    set_bounces(scene, 0)
    render_to(scene, args.out + "_direct.exr")
    print(f"[prism-reference] wrote {args.out}_full.exr / _direct.exr ({width}x{height}, Blender {bpy.app.version_string})")


main()
