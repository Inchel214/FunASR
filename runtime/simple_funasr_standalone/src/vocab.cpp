#include "vocab.h"
#include <fstream>
#include <sstream>

namespace funasr {

Vocab::Vocab(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Failed to open vocab file: " << filename << std::endl;
        return;
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    bool in_string = false;
    std::string token;
    for (size_t i = 0; i < content.size(); ++i) {
        char c = content[i];
        if (!in_string) {
            if (c == '\"') {
                in_string = true;
                token.clear();
            }
        } else {
            if (c == '\"') {
                in_string = false;
                vocab_.push_back(token);
            } else if (c == '\\') {
                if (i + 1 < content.size()) {
                    char next = content[i + 1];
                    token.push_back(next);
                    ++i;
                }
            } else {
                token.push_back(c);
            }
        }
    }
    loaded_ = !vocab_.empty();
}

Vocab::~Vocab() {}

std::string Vocab::Vector2String(const std::vector<int>& v) {
    std::string out;
    for (size_t i = 0; i < v.size(); i++) {
        int id = v[i];
        if (id < 0 || id >= (int)vocab_.size()) continue;
        std::string word = vocab_[id];
        if (word == "<s>" || word == "</s>" || word == "<blank>" || word == "<unk>") continue;
        if (word.size() >= 2 && word.substr(word.size() - 2) == "@@") {
            out += word.substr(0, word.size() - 2);
        } else {
            out += word;
        }
    }
    return out;
}

int Vocab::Size() const {
    return vocab_.size();
}

} // namespace funasr
