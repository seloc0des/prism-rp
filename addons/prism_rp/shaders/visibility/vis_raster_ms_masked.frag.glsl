#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_nonuniform_qualifier : require

#define PRISM_VIS_MESH_PATH
#define PRISM_VIS_MASKED

// Visibility-buffer fragment shader for vis_raster_masked.mesh.glsl (alpha test).
#include "vis_raster_frag_body.glsl"
