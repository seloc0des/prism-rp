#include "prism_rp/register_types.h"

#include <gdextension_interface.h>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace prism {

void initialize_prism_rp_module(godot::ModuleInitializationLevel p_level) {
    if (p_level != godot::MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    // Phase 1+ classes (ChronosVPUCompositorEffect, PrismTelemetry) register here via GDREGISTER_CLASS.
    godot::UtilityFunctions::print("[Prism RP] C++20 GDExtension initialized (Vulkan RenderingDevice backend ready).");
}

void uninitialize_prism_rp_module(godot::ModuleInitializationLevel p_level) {
    if (p_level != godot::MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
}

} // namespace prism

extern "C" {

GDExtensionBool GDE_EXPORT prism_rp_library_init(
    GDExtensionInterfaceGetProcAddress p_get_proc_address,
    const GDExtensionClassLibraryPtr p_library,
    GDExtensionInitialization *r_initialization
) {
    godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

    init_obj.register_initializer(prism::initialize_prism_rp_module);
    init_obj.register_terminator(prism::uninitialize_prism_rp_module);
    init_obj.set_minimum_library_initialization_level(godot::MODULE_INITIALIZATION_LEVEL_SCENE);

    return init_obj.init();
}

} // extern "C"
