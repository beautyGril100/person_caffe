#ifndef FD_TOOLS
#define FD_TOOLS

#include "retinaface_anchor_generator.h"

void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes);

#endif // FD_TOOLS
