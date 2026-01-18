#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cstring>

// 简单的日志宏替代 glog
#define LOG(level) std::cout << "[" << #level << "] "
#define VLOG(level) if(0) std::cout 

// NPU 替换标记
// TODO: Replace with NPU specific includes if needed
#include <onnxruntime_cxx_api.h>
