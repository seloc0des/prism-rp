#include "register_types.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/variant/variant.h"
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/rendering_device.h"

namespace {

const char *TRACK_B_SETTING = "rendering/prism_rp/track_b";

enum TrackBMode {
	TRACK_B_OFF,
	TRACK_B_AUTO,
};

// What the GPU offers Track B, taken from the RenderingDevice features the engine patches add.
struct PrismCapabilities {
	bool mesh_shader = false;
	bool task_shader = false;
	bool draw_indirect_count = false;

	// "mesh" needs mesh shaders and GPU-written draw counts. Without mesh shaders, geometry can still be
	// pulled in a vertex shader and drawn with indirect counts.
	String gpu_path() const {
		if (mesh_shader && draw_indirect_count) {
			return "mesh";
		}
		if (draw_indirect_count) {
			return "vertex-pulling";
		}
		return "unavailable";
	}
};

PrismCapabilities query_capabilities(RenderingDevice *p_device) {
	PrismCapabilities capabilities;
	capabilities.mesh_shader = p_device->has_feature(RenderingDevice::SUPPORTS_MESH_SHADER);
	capabilities.task_shader = p_device->has_feature(RenderingDevice::SUPPORTS_TASK_SHADER);
	capabilities.draw_indirect_count = p_device->has_feature(RenderingDevice::SUPPORTS_DRAW_INDIRECT_COUNT);
	return capabilities;
}

// Runs when the compositor creates the Forward+ scene renderer. No Track B renderer exists yet, so this only reports
// what it found and leaves the stock renderer in place by returning nullptr.
RendererSceneRenderRD *scene_renderer_factory() {
	TrackBMode mode = TrackBMode(int(GLOBAL_GET(TRACK_B_SETTING)));
	String source = "setting";
	for (const String &argument : OS::get_singleton()->get_cmdline_user_args()) {
		if (argument == "--prism-trackb=off") {
			mode = TRACK_B_OFF;
			source = "command line";
		} else if (argument == "--prism-trackb=auto") {
			mode = TRACK_B_AUTO;
			source = "command line";
		} else if (argument.begins_with("--prism-trackb=")) {
			WARN_PRINT("[Prism RP] Ignoring " + argument + ", expected --prism-trackb=off or --prism-trackb=auto.");
		}
	}

	const PrismCapabilities capabilities = query_capabilities(RenderingDevice::get_singleton());
	print_line(vformat("[Prism RP] Track B: %s (%s) | GPU path: %s", mode == TRACK_B_AUTO ? "auto" : "off", source, capabilities.gpu_path()));
	return nullptr;
}

} // namespace

// SERVERS runs before the display server and the rendering server are created, so the setting exists by the time the
// compositor asks for a scene renderer, and the factory is in place before it does. The RenderingDevice is only needed
// when the factory runs, which is after it exists.
void initialize_prism_rp_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		GLOBAL_DEF(PropertyInfo(Variant::INT, TRACK_B_SETTING, PROPERTY_HINT_ENUM, "Off,Auto"), TRACK_B_OFF);
		RendererCompositorRD::scene_renderer_factory = scene_renderer_factory;
	} else if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		print_line("[Prism RP] engine module loaded");
	}
}

void uninitialize_prism_rp_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		RendererCompositorRD::scene_renderer_factory = nullptr;
	}
}
