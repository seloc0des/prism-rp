#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_mesh_shader : require

#define PRISM_VIS_MESH_PATH

// Visibility-buffer fragment shader for vis_raster.mesh.glsl (opaque, early depth test).
#include "vis_raster_frag_body.glsl"
