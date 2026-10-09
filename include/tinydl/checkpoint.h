#pragma once

#include "tinydl/nn.h"

#include <string>

namespace tinydl {

void save_checkpoint(GptModel& model, const std::string& path);
void load_checkpoint(GptModel& model, const std::string& path);
GptConfig read_checkpoint_config(const std::string& path);

}  // namespace tinydl
