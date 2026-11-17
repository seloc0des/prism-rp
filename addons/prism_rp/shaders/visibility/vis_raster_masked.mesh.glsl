#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_mesh_shader : require

#define PRISM_VIS_MASKED

// Mesh-shader visibility-buffer rasterizer, alpha-tested list. Pair with vis_raster_ms_masked.frag.glsl.
#include "vis_raster_mesh_body.glsl"
