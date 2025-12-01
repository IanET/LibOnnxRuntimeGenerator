#define _Frees_ptr_opt_

// Assumes building on Win32 where ORT_CHAR_T gets defined as wchar_t. Redefine it back so we can switch at runtime
#define wchar_t ORT_CHAR_T

#include ".\microsoft.ml.onnxruntime\build\native\include\onnxruntime_c_api.h"
