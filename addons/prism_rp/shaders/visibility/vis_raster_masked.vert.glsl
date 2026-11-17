#version 460
#extension GL_GOOGLE_include_directive : require

#define PRISM_VIS_MASKED

// Vertex-pulling visibility-buffer rasterizer, alpha-tested list. Pair with vis_raster_masked.frag.glsl.
#include "vis_raster_vert_body.glsl"
