#pragma once

#include <godot_cpp/core/class_db.hpp>

namespace prism {

void initialize_prism_rp_module(godot::ModuleInitializationLevel p_level);
void uninitialize_prism_rp_module(godot::ModuleInitializationLevel p_level);

} // namespace prism
