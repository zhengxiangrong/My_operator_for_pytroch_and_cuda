
#include "dcn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dcn_forward", &dcn_forward, "dcn_forward");
  m.def("dcn_backward", &dcn_backward, "dcn_backward");
}
