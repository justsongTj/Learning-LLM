#include "tinydl/backend.h"

#include <stdexcept>

namespace tinydl {

#ifndef TINYDL_ENABLE_DCU
std::shared_ptr<Backend> make_dcu_backend(int) {
    throw std::runtime_error("TinyDL was built without TINYDL_ENABLE_DCU");
}

bool dcu_backend_available() noexcept { return false; }
#endif

}  // namespace tinydl
