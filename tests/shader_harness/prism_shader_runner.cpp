// Prism RP compute-shader test runner.
//
// Runs a sequence of compute dispatches on any Vulkan 1.3 device (in CI and dev containers that is
// Mesa's lavapipe, a CPU implementation) and dumps images and buffers back to disk, so shader math
// can be checked numerically without Godot or a GPU. Driven by a JSON job file written by
// tests/shader_harness/harness.py:
//
// {
//   "images":     { "<name>": { "format": "rgba16f", "width": 64, "height": 32, "mips": 1, "layers": 1,
//                               "array": false, "data": "in.bin" | null } },
//   "buffers":    { "<name>": { "size": 256, "data": "in.bin" | null } },
//   "dispatches": [ { "spv": "x.spv", "push_constants": "<hex>", "groups": [x, y, z],
//                     "indirect": { "buffer": "<name>", "offset": 0 },          (optional, replaces groups)
//                     "bindings": [ { "binding": 0, "image": "<name>", "kind": "sampled_nearest" },
//                                   { "binding": 1, "image": "<name>", "kind": "storage", "mip": 2 },
//                                   { "binding": 2, "images": ["a", "b"], "kind": "sampled_linear" },
//                                   { "binding": 3, "images": ["a"], "kind": "sampled_nearest", "array_size": 256 },
//                                     (array_size pads the list to a declared array size with its first image)
//                                   { "binding": 3, "buffer": "<name>", "kind": "storage_buffer" } ] } ],
//   "outputs":    { "<name>": "out.bin" }                                     (images or buffers)
// }
//
// kind: sampled_nearest | sampled_linear (combined image sampler, clamp to edge; linear is trilinear;
//       all mips, or just "mip" when given) | storage (one mip, all layers) | storage_buffer | uniform_buffer.
// Image data and readback hold every mip, mip-major then layer-major; image input may also be just
// mip 0 (the other mips start at zero). Missing data means zero-filled. Images stay in
// VK_IMAGE_LAYOUT_GENERAL for their whole life, so one image can be sampled in one dispatch and
// written in the next; a full memory barrier separates consecutive dispatches.
// With PRISM_VALIDATION=1 the Khronos validation layer is enabled and any error fails the run.

#include <vulkan/vulkan.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

int g_validation_errors = 0;

#define VK_CHECK(expr)                                                                         \
    do {                                                                                       \
        VkResult r_ = (expr);                                                                  \
        if (r_ != VK_SUCCESS) {                                                                \
            throw std::runtime_error(std::string(#expr) + " failed: " + std::to_string(r_));   \
        }                                                                                      \
    } while (0)

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT *data, void *) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        ++g_validation_errors;
        std::cerr << "VALIDATION ERROR: " << data->pMessage << "\n";
    }
    return VK_FALSE;
}

struct FormatInfo {
    VkFormat format;
    uint32_t texel_size;
};

FormatInfo format_info(const std::string &name) {
    static const std::map<std::string, FormatInfo> table = {
        { "r32f", { VK_FORMAT_R32_SFLOAT, 4 } },
        { "rg16f", { VK_FORMAT_R16G16_SFLOAT, 4 } },
        { "rgba16f", { VK_FORMAT_R16G16B16A16_SFLOAT, 8 } },
        { "rgba32f", { VK_FORMAT_R32G32B32A32_SFLOAT, 16 } },
        { "r8", { VK_FORMAT_R8_UNORM, 1 } },
        { "rgba8", { VK_FORMAT_R8G8B8A8_UNORM, 4 } },
        { "r32ui", { VK_FORMAT_R32_UINT, 4 } },
        { "rg32ui", { VK_FORMAT_R32G32_UINT, 8 } },
    };
    auto it = table.find(name);
    if (it == table.end()) {
        throw std::runtime_error("unknown image format: " + name);
    }
    return it->second;
}

std::vector<char> read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<char> data(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(data.data(), static_cast<std::streamsize>(data.size()));
    return data;
}

std::vector<uint8_t> hex_to_bytes(const std::string &hex) {
    std::vector<uint8_t> out;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        out.push_back(static_cast<uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
    }
    return out;
}

struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkPhysicalDeviceMemoryProperties mem_props{};

    uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) const {
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & flags) == flags) {
                return i;
            }
        }
        throw std::runtime_error("no suitable memory type");
    }
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view_all = VK_NULL_HANDLE;        // every mip and layer (sampled bindings)
    std::vector<VkImageView> mip_views;           // one per mip, every layer (storage bindings)
    FormatInfo info{};
    uint32_t width = 0, height = 0, mips = 1, layers = 1;
    bool array = false;
    Buffer staging;   // host-visible, used for upload and readback

    uint32_t mip_width(uint32_t m) const { return std::max(1u, width >> m); }
    uint32_t mip_height(uint32_t m) const { return std::max(1u, height >> m); }
    VkDeviceSize mip_bytes(uint32_t m) const {
        return static_cast<VkDeviceSize>(mip_width(m)) * mip_height(m) * layers * info.texel_size;
    }
    VkDeviceSize total_bytes() const {
        VkDeviceSize total = 0;
        for (uint32_t m = 0; m < mips; ++m) total += mip_bytes(m);
        return total;
    }
    std::vector<VkBufferImageCopy> regions() const {
        std::vector<VkBufferImageCopy> out;
        VkDeviceSize offset = 0;
        for (uint32_t m = 0; m < mips; ++m) {
            VkBufferImageCopy r{};
            r.bufferOffset = offset;
            r.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, m, 0, layers };
            r.imageExtent = { mip_width(m), mip_height(m), 1 };
            out.push_back(r);
            offset += mip_bytes(m);
        }
        return out;
    }
};

struct DeviceBuffer {
    Buffer device;
    Buffer staging;
};

Buffer make_buffer(const Context &ctx, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) {
    Buffer b;
    b.size = size;
    VkBufferCreateInfo ci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    ci.size = size;
    ci.usage = usage;
    VK_CHECK(vkCreateBuffer(ctx.device, &ci, nullptr, &b.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx.device, b.buffer, &req);
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = ctx.memory_type(req.memoryTypeBits, flags);
    VK_CHECK(vkAllocateMemory(ctx.device, &ai, nullptr, &b.memory));
    VK_CHECK(vkBindBufferMemory(ctx.device, b.buffer, b.memory, 0));
    return b;
}

Buffer make_host_buffer(const Context &ctx, VkDeviceSize size) {
    return make_buffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void destroy_buffer(const Context &ctx, Buffer &b) {
    vkDestroyBuffer(ctx.device, b.buffer, nullptr);
    vkFreeMemory(ctx.device, b.memory, nullptr);
}

void init_context(Context &ctx) {
    const bool want_validation = std::getenv("PRISM_VALIDATION") != nullptr && std::string(std::getenv("PRISM_VALIDATION")) == "1";

    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "prism_shader_runner";
    app.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char *> layers;
    std::vector<const char *> extensions;
    if (want_validation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = static_cast<uint32_t>(layers.size());
    ici.ppEnabledLayerNames = layers.data();
    ici.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ici.ppEnabledExtensionNames = extensions.data();
    VK_CHECK(vkCreateInstance(&ici, nullptr, &ctx.instance));

    if (want_validation) {
        auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(ctx.instance, "vkCreateDebugUtilsMessengerEXT"));
        VkDebugUtilsMessengerCreateInfoEXT mci{ VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
        mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
        mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
        mci.pfnUserCallback = debug_callback;
        if (create != nullptr) {
            VK_CHECK(create(ctx.instance, &mci, nullptr, &ctx.messenger));
        }
    }

    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &count, nullptr));
    if (count == 0) {
        throw std::runtime_error("no Vulkan devices (install mesa-vulkan-drivers for lavapipe)");
    }
    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &count, devices.data()));
    ctx.physical = devices[0];
    vkGetPhysicalDeviceMemoryProperties(ctx.physical, &ctx.mem_props);

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical, &qf_count, qfs.data());
    bool found = false;
    for (uint32_t i = 0; i < qf_count; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx.queue_family = i;
            found = true;
            break;
        }
    }
    if (!found) {
        throw std::runtime_error("no compute queue");
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = ctx.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    // Texture arrays indexed with nonuniformEXT (the Track B material passes) need descriptor
    // indexing; enable what the device has.
    VkPhysicalDeviceVulkan12Features supported12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceFeatures2 supported{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    supported.pNext = &supported12;
    vkGetPhysicalDeviceFeatures2(ctx.physical, &supported);

    VkPhysicalDeviceVulkan12Features features12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.runtimeDescriptorArray = supported12.runtimeDescriptorArray;
    features12.shaderSampledImageArrayNonUniformIndexing = supported12.shaderSampledImageArrayNonUniformIndexing;
    VkPhysicalDeviceFeatures2 features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    features.pNext = &features12;
    // Godot enables these for storage images written without a format qualifier match check.
    features.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    features.features.shaderStorageImageReadWithoutFormat = VK_TRUE;

    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &features;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VK_CHECK(vkCreateDevice(ctx.physical, &dci, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queue_family, 0, &ctx.queue);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx.physical, &props);
    std::cerr << "device: " << props.deviceName << "\n";
}

VkImageView make_view(const Context &ctx, const Image &img, uint32_t base_mip, uint32_t mip_count) {
    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = img.image;
    vci.viewType = img.array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    vci.format = img.info.format;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, base_mip, mip_count, 0, img.layers };
    VkImageView view;
    VK_CHECK(vkCreateImageView(ctx.device, &vci, nullptr, &view));
    return view;
}

Image make_image(const Context &ctx, const json &spec) {
    Image img;
    img.info = format_info(spec.at("format"));
    img.width = spec.at("width");
    img.height = spec.at("height");
    img.mips = spec.value("mips", 1u);
    img.layers = spec.value("layers", 1u);
    img.array = spec.value("array", img.layers > 1);

    VkImageCreateInfo ci{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = img.info.format;
    ci.extent = { img.width, img.height, 1 };
    ci.mipLevels = img.mips;
    ci.arrayLayers = img.layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(ctx.device, &ci, nullptr, &img.image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(ctx.device, img.image, &req);
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = ctx.memory_type(req.memoryTypeBits, 0);
    VK_CHECK(vkAllocateMemory(ctx.device, &ai, nullptr, &img.memory));
    VK_CHECK(vkBindImageMemory(ctx.device, img.image, img.memory, 0));

    img.view_all = make_view(ctx, img, 0, img.mips);
    for (uint32_t m = 0; m < img.mips; ++m) {
        img.mip_views.push_back(make_view(ctx, img, m, 1));
    }
    img.staging = make_host_buffer(ctx, img.total_bytes());
    return img;
}

VkSampler make_sampler(const Context &ctx, VkFilter filter) {
    VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter = filter;
    sci.minFilter = filter;
    sci.mipmapMode = filter == VK_FILTER_LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;   // as Godot's samplers
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = VK_LOD_CLAMP_NONE;
    VkSampler s;
    VK_CHECK(vkCreateSampler(ctx.device, &sci, nullptr, &s));
    return s;
}

void full_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
            VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

// Copies a data file into a mapped staging buffer. `allowed_short` is a smaller size that is also
// accepted (an image's mip 0); anything not covered stays zero.
void fill_staging(const Context &ctx, const Buffer &staging, const json &spec, const std::string &what, VkDeviceSize allowed_short) {
    void *mapped = nullptr;
    VK_CHECK(vkMapMemory(ctx.device, staging.memory, 0, staging.size, 0, &mapped));
    std::memset(mapped, 0, staging.size);
    if (spec.contains("data") && !spec.at("data").is_null()) {
        const std::vector<char> data = read_file(spec.at("data"));
        if (data.size() != staging.size && data.size() != allowed_short) {
            throw std::runtime_error(what + ": data is " + std::to_string(data.size()) + " bytes, expected " +
                    std::to_string(staging.size));
        }
        std::memcpy(mapped, data.data(), data.size());
    }
    vkUnmapMemory(ctx.device, staging.memory);
}

int run(const json &job) {
    Context ctx;
    init_context(ctx);

    VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.queueFamilyIndex = ctx.queue_family;
    VkCommandPool pool;
    VK_CHECK(vkCreateCommandPool(ctx.device, &pci, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cai, &cmd));
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    const VkSampler nearest = make_sampler(ctx, VK_FILTER_NEAREST);
    const VkSampler linear = make_sampler(ctx, VK_FILTER_LINEAR);

    // Images: create, move to GENERAL, upload or clear.
    std::map<std::string, Image> images;
    for (const auto &[name, spec] : job.at("images").items()) {
        Image img = make_image(ctx, spec);
        fill_staging(ctx, img.staging, spec, "image '" + name + "'", img.mip_bytes(0));

        VkImageMemoryBarrier ib{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image = img.image;
        ib.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, img.mips, 0, img.layers };
        ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);

        const std::vector<VkBufferImageCopy> regions = img.regions();
        vkCmdCopyBufferToImage(cmd, img.staging.buffer, img.image, VK_IMAGE_LAYOUT_GENERAL,
                static_cast<uint32_t>(regions.size()), regions.data());
        images.emplace(name, img);
    }

    // Buffers: create, upload or clear.
    std::map<std::string, DeviceBuffer> buffers;
    if (job.contains("buffers")) {
        for (const auto &[name, spec] : job.at("buffers").items()) {
            const VkDeviceSize size = spec.at("size");
            DeviceBuffer b;
            b.device = make_buffer(ctx, size,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0);
            b.staging = make_host_buffer(ctx, size);
            fill_staging(ctx, b.staging, spec, "buffer '" + name + "'", size);
            VkBufferCopy region{ 0, 0, size };
            vkCmdCopyBuffer(cmd, b.staging.buffer, b.device.buffer, 1, &region);
            buffers.emplace(name, b);
        }
    }
    full_barrier(cmd);

    // Dispatches.
    std::vector<VkShaderModule> modules;
    std::vector<VkPipeline> pipelines;
    std::vector<VkPipelineLayout> layouts;
    std::vector<VkDescriptorSetLayout> set_layouts;
    std::vector<VkDescriptorPool> pools;

    for (const json &d : job.at("dispatches")) {
        const std::vector<char> spv = read_file(d.at("spv"));
        VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smci.codeSize = spv.size();
        smci.pCode = reinterpret_cast<const uint32_t *>(spv.data());
        VkShaderModule module;
        VK_CHECK(vkCreateShaderModule(ctx.device, &smci, nullptr, &module));
        modules.push_back(module);

        const json &binds = d.at("bindings");
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        std::vector<std::vector<VkDescriptorImageInfo>> image_infos(binds.size());
        std::vector<VkDescriptorBufferInfo> buffer_infos(binds.size());
        std::map<VkDescriptorType, uint32_t> type_counts;
        for (size_t i = 0; i < binds.size(); ++i) {
            const json &b = binds[i];
            const std::string kind = b.at("kind");
            VkDescriptorSetLayoutBinding lb{};
            lb.binding = b.at("binding");
            lb.descriptorCount = 1;
            lb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            if (kind == "storage_buffer" || kind == "uniform_buffer") {
                lb.descriptorType = (kind == "storage_buffer") ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                const DeviceBuffer &buf = buffers.at(b.at("buffer").get<std::string>());
                buffer_infos[i] = { buf.device.buffer, 0, VK_WHOLE_SIZE };
            } else {
                std::vector<std::string> names;
                if (b.contains("images")) {
                    names = b.at("images").get<std::vector<std::string>>();
                } else {
                    names.push_back(b.at("image").get<std::string>());
                }
                // An array binding is declared with a fixed size; fill the unused slots with its first image.
                if (b.contains("array_size")) {
                    const size_t array_size = b.at("array_size").get<size_t>();
                    if (names.size() > array_size) {
                        throw std::runtime_error("binding " + std::to_string(lb.binding) + " has " + std::to_string(names.size()) + " images for an array of " + std::to_string(array_size));
                    }
                    names.resize(array_size, names.at(0));
                }
                lb.descriptorCount = static_cast<uint32_t>(names.size());
                for (const std::string &n : names) {
                    const Image &img = images.at(n);
                    VkDescriptorImageInfo info{};
                    info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    if (kind == "storage") {
                        lb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                        info.imageView = img.mip_views.at(b.value("mip", 0u));
                    } else if (kind == "sampled_nearest" || kind == "sampled_linear") {
                        lb.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                        info.imageView = b.contains("mip") ? img.mip_views.at(b.at("mip").get<uint32_t>()) : img.view_all;
                        info.sampler = (kind == "sampled_linear") ? linear : nearest;
                    } else {
                        throw std::runtime_error("unknown binding kind: " + kind);
                    }
                    image_infos[i].push_back(info);
                }
            }
            type_counts[lb.descriptorType] += lb.descriptorCount;
            bindings.push_back(lb);
        }

        VkDescriptorSetLayoutCreateInfo dslci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        dslci.bindingCount = static_cast<uint32_t>(bindings.size());
        dslci.pBindings = bindings.data();
        VkDescriptorSetLayout set_layout;
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &dslci, nullptr, &set_layout));
        set_layouts.push_back(set_layout);

        const std::vector<uint8_t> pc = hex_to_bytes(d.value("push_constants", std::string()));
        VkPushConstantRange range{ VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(pc.size()) };
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &set_layout;
        plci.pushConstantRangeCount = pc.empty() ? 0 : 1;
        plci.pPushConstantRanges = &range;
        VkPipelineLayout layout;
        VK_CHECK(vkCreatePipelineLayout(ctx.device, &plci, nullptr, &layout));
        layouts.push_back(layout);

        VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cpci.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = module;
        cpci.stage.pName = "main";
        cpci.layout = layout;
        VkPipeline pipeline;
        VK_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline));
        pipelines.push_back(pipeline);

        std::vector<VkDescriptorPoolSize> sizes;
        for (const auto &[type, n] : type_counts) sizes.push_back({ type, n });
        VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        dpci.maxSets = 1;
        dpci.poolSizeCount = static_cast<uint32_t>(sizes.size());
        dpci.pPoolSizes = sizes.data();
        VkDescriptorPool dpool;
        VK_CHECK(vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &dpool));
        pools.push_back(dpool);

        VkDescriptorSetAllocateInfo dsai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dsai.descriptorPool = dpool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &set_layout;
        VkDescriptorSet set;
        VK_CHECK(vkAllocateDescriptorSets(ctx.device, &dsai, &set));

        std::vector<VkWriteDescriptorSet> writes;
        for (size_t i = 0; i < bindings.size(); ++i) {
            VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            w.dstSet = set;
            w.dstBinding = bindings[i].binding;
            w.descriptorCount = bindings[i].descriptorCount;
            w.descriptorType = bindings[i].descriptorType;
            if (image_infos[i].empty()) {
                w.pBufferInfo = &buffer_infos[i];
            } else {
                w.pImageInfo = image_infos[i].data();
            }
            writes.push_back(w);
        }
        vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        if (!pc.empty()) {
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(pc.size()), pc.data());
        }
        if (d.contains("indirect") && !d.at("indirect").is_null()) {
            const json &ind = d.at("indirect");
            vkCmdDispatchIndirect(cmd, buffers.at(ind.at("buffer").get<std::string>()).device.buffer, ind.value("offset", 0u));
        } else {
            const json &g = d.at("groups");
            vkCmdDispatch(cmd, g.at(0), g.at(1), g.at(2));
        }
        full_barrier(cmd);
    }

    // Readback.
    for (const auto &[name, path] : job.at("outputs").items()) {
        if (images.count(name)) {
            const Image &img = images.at(name);
            const std::vector<VkBufferImageCopy> regions = img.regions();
            vkCmdCopyImageToBuffer(cmd, img.image, VK_IMAGE_LAYOUT_GENERAL, img.staging.buffer,
                    static_cast<uint32_t>(regions.size()), regions.data());
        } else {
            const DeviceBuffer &b = buffers.at(name);
            VkBufferCopy region{ 0, 0, b.device.size };
            vkCmdCopyBuffer(cmd, b.device.buffer, b.staging.buffer, 1, &region);
        }
    }
    VkMemoryBarrier hb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    VK_CHECK(vkCreateFence(ctx.device, &fci, nullptr, &fence));
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(ctx.queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX));

    for (const auto &[name, path] : job.at("outputs").items()) {
        const Buffer &staging = images.count(name) ? images.at(name).staging : buffers.at(name).staging;
        void *mapped = nullptr;
        VK_CHECK(vkMapMemory(ctx.device, staging.memory, 0, staging.size, 0, &mapped));
        std::ofstream out(path.get<std::string>(), std::ios::binary);
        out.write(static_cast<const char *>(mapped), static_cast<std::streamsize>(staging.size));
        vkUnmapMemory(ctx.device, staging.memory);
    }

    // The process exits right after; the OS reclaims everything. Tearing down the device keeps
    // the validation layer from reporting leaks, so do it anyway.
    vkDeviceWaitIdle(ctx.device);
    vkDestroyFence(ctx.device, fence, nullptr);
    for (auto p : pipelines) vkDestroyPipeline(ctx.device, p, nullptr);
    for (auto l : layouts) vkDestroyPipelineLayout(ctx.device, l, nullptr);
    for (auto l : set_layouts) vkDestroyDescriptorSetLayout(ctx.device, l, nullptr);
    for (auto p : pools) vkDestroyDescriptorPool(ctx.device, p, nullptr);
    for (auto m : modules) vkDestroyShaderModule(ctx.device, m, nullptr);
    for (auto &[name, img] : images) {
        vkDestroyImageView(ctx.device, img.view_all, nullptr);
        for (auto v : img.mip_views) vkDestroyImageView(ctx.device, v, nullptr);
        vkDestroyImage(ctx.device, img.image, nullptr);
        vkFreeMemory(ctx.device, img.memory, nullptr);
        destroy_buffer(ctx, img.staging);
    }
    for (auto &[name, b] : buffers) {
        destroy_buffer(ctx, b.device);
        destroy_buffer(ctx, b.staging);
    }
    vkDestroySampler(ctx.device, nearest, nullptr);
    vkDestroySampler(ctx.device, linear, nullptr);
    vkDestroyCommandPool(ctx.device, pool, nullptr);
    vkDestroyDevice(ctx.device, nullptr);
    if (ctx.messenger != VK_NULL_HANDLE) {
        auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(ctx.instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy) destroy(ctx.instance, ctx.messenger, nullptr);
    }
    vkDestroyInstance(ctx.instance, nullptr);

    if (g_validation_errors > 0) {
        std::cerr << g_validation_errors << " validation error(s)\n";
        return 3;
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: prism_shader_runner job.json\n";
        return 2;
    }
    try {
        std::ifstream f(argv[1]);
        const json job = json::parse(f);
        return run(job);
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
