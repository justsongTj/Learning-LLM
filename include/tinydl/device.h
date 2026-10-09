#pragma once

namespace tinydl {

enum class DeviceType { CPU, DCU };

struct Device {
    DeviceType type{DeviceType::CPU};
    int index{0};

    static Device cpu() { return {}; }
    static Device dcu(int index = 0) { return {DeviceType::DCU, index}; }
};

}  // namespace tinydl
