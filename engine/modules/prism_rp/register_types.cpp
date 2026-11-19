#include "register_types.h"

#include "core/string/print_string.h"
#include "core/variant/variant.h"

// SCENE is the first level where the servers and RenderingDevice exist, which
// is what the later Track B hooks need. Nothing earlier is required.
void initialize_prism_rp_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	print_line("[Prism RP] engine module loaded");
}

void uninitialize_prism_rp_module(ModuleInitializationLevel p_level) {
}
