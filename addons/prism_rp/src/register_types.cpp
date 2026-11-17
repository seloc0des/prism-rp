#include "prism_rp/register_types.h"
#include "prism_rp/chronos_vpu_effect.h"
#include "prism_rp/prism_telemetry.h"

#include <gdextension_interface.h>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/core/memory.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace prism {

static PrismTelemetry *g_telemetry_singleton = nullptr;

void initialize_prism_rp_module(godot::ModuleInitializationLevel p_level) {
    if (p_level != godot::MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    // Abstract: only this file creates the instance. Scripts and the editor (which instantiates
    // classes to read default property values) can't spawn copies that steal the singleton.
    GDREGISTER_ABSTRACT_CLASS(PrismTelemetry);
    GDREGISTER_CLASS(ChronosVPUCompositorEffect);

    g_telemetry_singleton = memnew(PrismTelemetry);
    PrismTelemetry::set_singleton(g_telemetry_singleton);
    godot::Engine::get_singleton()->register_singleton("PrismTelemetry", g_telemetry_singleton);

    godot::UtilityFunctions::print("[Prism RP] Registered PrismTelemetry & ChronosVPUCompositorEffect (Vulkan RenderingDevice ready).");
}

void uninitialize_prism_rp_module(godot::ModuleInitializationLevel p_level) {
    if (p_level != godot::MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    if (g_telemetry_singleton != nullptr) {
        godot::Engine::get_singleton()->unregister_singleton("PrismTelemetry");
        PrismTelemetry::set_singleton(nullptr);
        memdelete(g_telemetry_singleton);
        g_telemetry_singleton = nullptr;
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
