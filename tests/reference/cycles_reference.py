"""Render a Prism RP scene export (project/benchmarks/prism_scene_export.gd) with Cycles.

    blender -b --factory-startup --python tests/reference/cycles_reference.py -- \
        --scene out/scene.json --out out/reference --samples 256 --diffuse-bounces 1

Writes <out>_diffuse_indirect.npy (linear float32 HxWx3, top row first), the Cycles "Diffuse
Indirect" pass, i.e. bounced diffuse light for white albedo, which is the quantity Lumina's
"Denoised Irradiance" debug view shows; and <out>_combined.png for a visual check.
"""

import argparse
import json
import math
import os
import sys

from array import array

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
        mix = nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = 1.0
        mix.inputs["Color2"].default_value = (r, g, b, 1.0)
        links.new(tex.outputs["Color"], mix.inputs["Color1"])
        links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
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

    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], composite.inputs["Image"])
    # The Viewer node's pixels aren't populated in background mode, so the pass goes to a float EXR
    # that is read back after the render.
    out = tree.nodes.new("CompositorNodeOutputFile")
    out.base_path = os.path.dirname(os.path.abspath(args.out))
    out.format.file_format = "OPEN_EXR"
    out.format.color_depth = "32"
    out.format.color_management = "OVERRIDE"
    out.format.linear_colorspace_settings.name = "Linear Rec.709"
    out.file_slots[0].path = os.path.basename(args.out) + "_diffind_"
    tree.links.new(rl.outputs["DiffInd"], out.inputs[0])


def write_npy(path, data, shape):
    """Minimal .npy writer (Blender's bundled Python may not have numpy)."""
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }" % (tuple(shape),)
    header = header + " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00")
        f.write(len(header).to_bytes(2, "little"))
        f.write(header.encode("latin1"))
        data.tofile(f)


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

    bpy.context.scene.render.filepath = args.out + "_combined.png"
    bpy.ops.render.render(write_still=True)

    exr_path = f"{args.out}_diffind_{bpy.context.scene.frame_current:04d}.exr"
    img = bpy.data.images.load(exr_path)
    img.colorspace_settings.name = "Non-Color"   # keep the stored linear values untouched
    w, h = img.size[0], img.size[1]
    rgba = img.pixels[:]
    rgb = array("f")
    for row in range(h - 1, -1, -1):   # Blender rows are bottom-up; write top row first
        base = row * w * 4
        for x in range(w):
            i = base + x * 4
            rgb.extend(rgba[i:i + 3])
    write_npy(args.out + "_diffuse_indirect.npy", rgb, (h, w, 3))
    os.remove(exr_path)
    print(f"[prism-reference] wrote {args.out}_diffuse_indirect.npy ({w}x{h})")


main()
