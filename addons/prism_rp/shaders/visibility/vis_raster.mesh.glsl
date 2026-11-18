#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_mesh_shader : require

// Mesh-shader visibility-buffer rasterizer, opaque list. Pair with vis_raster_ms.frag.glsl.
#include "vis_raster_mesh_body.glsl"
