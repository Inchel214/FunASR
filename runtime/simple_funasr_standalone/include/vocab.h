#pragma once
#include "precomp.h"

namespace funasr {

class Vocab {
  public:
    Vocab(const std::string& filename);
    ~Vocab();
    std::string Vector2String(const std::vector<int>& v);
    int Size() const;
    bool IsLoaded() const { return loaded_; }

  private:
    std::vector<std::string> vocab_;
    bool loaded_ = false;
};

} // namespace funasr
