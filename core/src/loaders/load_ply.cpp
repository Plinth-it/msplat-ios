#include "loaders.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace {

// The descriptor stores point indices as int32 elsewhere.
constexpr uint64_t kMaxPlyVertices =
    static_cast<uint64_t>(std::numeric_limits<int32_t>::max());

// Property types in PLY files
enum class PlyType { Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32, Float64 };

bool parsePlyType(const std::string &t, PlyType &out) {
    if (t == "char" || t == "int8") out = PlyType::Int8;
    else if (t == "uchar" || t == "uint8") out = PlyType::UInt8;
    else if (t == "short" || t == "int16") out = PlyType::Int16;
    else if (t == "ushort" || t == "uint16") out = PlyType::UInt16;
    else if (t == "int" || t == "int32") out = PlyType::Int32;
    else if (t == "uint" || t == "uint32") out = PlyType::UInt32;
    else if (t == "float" || t == "float32") out = PlyType::Float32;
    else if (t == "double" || t == "float64") out = PlyType::Float64;
    else return false;
    return true;
}

size_t plyTypeSize(PlyType t) {
    switch (t) {
        case PlyType::Int8:
        case PlyType::UInt8:   return 1;
        case PlyType::Int16:
        case PlyType::UInt16:  return 2;
        case PlyType::Int32:
        case PlyType::UInt32:
        case PlyType::Float32: return 4;
        case PlyType::Float64: return 8;
    }
    return 0;
}

bool isPlyFloat(PlyType t) {
    return t == PlyType::Float32 || t == PlyType::Float64;
}

struct PlyProp {
    std::string name;
    PlyType type = PlyType::Float32;
    bool isList = false;
    PlyType listCountType = PlyType::UInt8;
    size_t offset = 0;  // byte offset within a binary vertex record
};

struct PlyElement {
    std::string name;
    uint64_t count = 0;
    std::vector<PlyProp> props;
};

[[noreturn]] void malformedPly(const std::string &path, const std::string &reason) {
    throw std::runtime_error("Malformed PLY file " + path + ": " + reason);
}

std::string stripCarriageReturn(std::string line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return line;
}

bool parseCount(const std::string &token, uint64_t &out) {
    if (token.empty() || token[0] < '0' || token[0] > '9') return false;
    errno = 0;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(token.c_str(), &end, 10);
    if (errno != 0 || end != token.c_str() + token.size()) return false;
    out = static_cast<uint64_t>(value);
    return true;
}

double readBinaryScalar(const char *bytes, PlyType type) {
    switch (type) {
        case PlyType::Int8:    { int8_t v;   memcpy(&v, bytes, 1); return v; }
        case PlyType::UInt8:   { uint8_t v;  memcpy(&v, bytes, 1); return v; }
        case PlyType::Int16:   { int16_t v;  memcpy(&v, bytes, 2); return v; }
        case PlyType::UInt16:  { uint16_t v; memcpy(&v, bytes, 2); return v; }
        case PlyType::Int32:   { int32_t v;  memcpy(&v, bytes, 4); return v; }
        case PlyType::UInt32:  { uint32_t v; memcpy(&v, bytes, 4); return v; }
        case PlyType::Float32: { float v;    memcpy(&v, bytes, 4); return v; }
        case PlyType::Float64: { double v;   memcpy(&v, bytes, 8); return v; }
    }
    return 0.0;
}

// Floats are normalized [0, 1] colors; 16-bit channels keep their high byte.
uint8_t colorByte(double value, PlyType type) {
    if (std::isnan(value)) return 0;
    if (isPlyFloat(type)) value *= 255.0;
    else if (type == PlyType::UInt16) value /= 256.0;
    return static_cast<uint8_t>(std::lround(std::clamp(value, 0.0, 255.0)));
}

// Skips one element that precedes the vertices. List properties make binary
// records variable-length, so each record is walked.
void skipElement(std::ifstream &f, const PlyElement &element, bool binary,
                 uint64_t &remaining, const std::string &path) {
    if (!binary) {
        // Each ASCII record occupies at least one byte (its newline).
        if (element.count > remaining)
            malformedPly(path, "element " + element.name + " count exceeds the file size");
        std::string line;
        for (uint64_t i = 0; i < element.count; i++) {
            if (!std::getline(f, line))
                malformedPly(path, "truncated element " + element.name);
            remaining -= std::min<uint64_t>(remaining, line.size() + 1);
        }
        return;
    }

    uint64_t minimumRecordBytes = 0;
    for (const auto &prop : element.props)
        minimumRecordBytes += plyTypeSize(prop.isList ? prop.listCountType : prop.type);
    if (minimumRecordBytes == 0) {
        if (element.count != 0)
            malformedPly(path, "element " + element.name + " has records but no properties");
        return;
    }
    if (element.count > remaining / minimumRecordBytes)
        malformedPly(path, "element " + element.name + " count exceeds the file size");

    char scalar[8];
    auto take = [&](size_t bytes, char *destination) {
        if (bytes > remaining) malformedPly(path, "truncated element " + element.name);
        if (destination) {
            f.read(destination, static_cast<std::streamsize>(bytes));
        } else {
            f.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
        }
        if (!f) malformedPly(path, "truncated element " + element.name);
        remaining -= bytes;
    };
    for (uint64_t i = 0; i < element.count; i++) {
        for (const auto &prop : element.props) {
            if (!prop.isList) {
                take(plyTypeSize(prop.type), nullptr);
                continue;
            }
            const size_t countBytes = plyTypeSize(prop.listCountType);
            take(countBytes, scalar);
            const double length = readBinaryScalar(scalar, prop.listCountType);
            if (length < 0 || length > static_cast<double>(remaining))
                malformedPly(path, "invalid list length in element " + element.name);
            const uint64_t itemBytes = plyTypeSize(prop.type);
            const uint64_t items = static_cast<uint64_t>(length);
            if (items > remaining / itemBytes)
                malformedPly(path, "truncated element " + element.name);
            take(static_cast<size_t>(items * itemBytes), nullptr);
        }
    }
}

} // namespace

SparsePointSet readPly(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open PLY file: " + path);

    std::string line;
    if (!std::getline(f, line) || stripCarriageReturn(line).rfind("ply", 0) != 0)
        throw std::runtime_error("Not a PLY file: " + path);

    bool binary = false;
    bool ascii = false;
    bool headerEnded = false;
    std::vector<PlyElement> elements;

    while (std::getline(f, line)) {
        line = stripCarriageReturn(line);
        if (line == "end_header") {
            headerEnded = true;
            break;
        }

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "format") {
            std::string fmt;
            iss >> fmt;
            binary = (fmt == "binary_little_endian");
            ascii = (fmt == "ascii");
        } else if (token == "element") {
            PlyElement element;
            std::string countToken;
            iss >> element.name >> countToken;
            if (!parseCount(countToken, element.count))
                malformedPly(path, "invalid count for element " + element.name);
            elements.push_back(std::move(element));
        } else if (token == "property") {
            if (elements.empty()) malformedPly(path, "property declared before any element");
            PlyProp prop;
            std::string typeStr;
            iss >> typeStr;
            if (typeStr == "list") {
                std::string countType, itemType;
                iss >> countType >> itemType;
                if (!parsePlyType(countType, prop.listCountType) ||
                    !parsePlyType(itemType, prop.type) ||
                    isPlyFloat(prop.listCountType))
                    malformedPly(path, "unsupported list property type");
                prop.isList = true;
            } else if (!parsePlyType(typeStr, prop.type)) {
                malformedPly(path, "unsupported property type " + typeStr);
            }
            iss >> prop.name;
            elements.back().props.push_back(std::move(prop));
        }
    }

    if (!headerEnded) malformedPly(path, "missing end_header");
    if (!binary && !ascii) throw std::runtime_error("Unsupported PLY format (need ascii or binary_little_endian)");

    const auto vertexIt = std::find_if(elements.begin(), elements.end(),
        [](const PlyElement &element) { return element.name == "vertex"; });
    if (vertexIt == elements.end() || vertexIt->count == 0)
        throw std::runtime_error("PLY has no vertices");
    const PlyElement &vertex = *vertexIt;
    if (vertex.count > kMaxPlyVertices)
        malformedPly(path, "vertex count is outside the supported range");

    std::vector<PlyProp> props = vertex.props;
    size_t vertexBytes = 0;
    for (auto &prop : props) {
        if (prop.isList) malformedPly(path, "list properties are not supported on vertices");
        prop.offset = vertexBytes;
        vertexBytes += plyTypeSize(prop.type);
    }

    // Find property indices
    auto findProp = [&](const std::string &name) -> int {
        for (int i = 0; i < (int)props.size(); i++)
            if (props[i].name == name) return i;
        return -1;
    };

    int ix = findProp("x"), iy = findProp("y"), iz = findProp("z");
    int ir = findProp("red"), ig = findProp("green"), ib = findProp("blue");
    if (ix < 0 || iy < 0 || iz < 0)
        throw std::runtime_error("PLY missing x/y/z properties");
    const bool hasColor = ir >= 0 && ig >= 0 && ib >= 0;

    const std::streampos dataStart = f.tellg();
    f.seekg(0, std::ios::end);
    const std::streampos fileEnd = f.tellg();
    f.seekg(dataStart);
    if (dataStart < 0 || fileEnd < dataStart || !f)
        malformedPly(path, "cannot determine the payload size");
    uint64_t remaining = static_cast<uint64_t>(fileEnd - dataStart);

    for (auto it = elements.begin(); it != vertexIt; ++it)
        skipElement(f, *it, binary, remaining, path);

    // Bound the allocation by the bytes that could hold the records: a binary
    // record is fixed-size, and an ASCII value needs a digit and a separator.
    const uint64_t minimumRecordBytes = binary ? vertexBytes : props.size() * 2;
    if (vertex.count > remaining / minimumRecordBytes)
        malformedPly(path, "vertex count exceeds the file size");
    const size_t numVertices = static_cast<size_t>(vertex.count);

    SparsePointSet pts;
    pts.xyz.resize(numVertices * 3);
    pts.rgb.resize(numVertices * 3, 128); // default gray if no color

    if (binary) {
        std::vector<char> buf(vertexBytes);
        auto value = [&](int pi) {
            return readBinaryScalar(&buf[props[pi].offset], props[pi].type);
        };
        for (size_t i = 0; i < numVertices; i++) {
            if (!f.read(buf.data(), static_cast<std::streamsize>(vertexBytes)))
                throw std::runtime_error("Truncated binary PLY vertex " +
                                         std::to_string(i) + ": " + path);

            pts.xyz[i*3+0] = static_cast<float>(value(ix));
            pts.xyz[i*3+1] = static_cast<float>(value(iy));
            pts.xyz[i*3+2] = static_cast<float>(value(iz));

            if (hasColor) {
                pts.rgb[i*3+0] = colorByte(value(ir), props[ir].type);
                pts.rgb[i*3+1] = colorByte(value(ig), props[ig].type);
                pts.rgb[i*3+2] = colorByte(value(ib), props[ib].type);
            }
        }
    } else {
        // ASCII
        std::vector<std::string> tokens;
        auto value = [&](int pi, size_t i) {
            const std::string &token = tokens[pi];
            char *end = nullptr;
            const double parsed = std::strtod(token.c_str(), &end);
            if (end != token.c_str() + token.size())
                throw std::runtime_error("Invalid ASCII PLY value in vertex " +
                                         std::to_string(i) + ": " + path);
            return parsed;
        };
        for (size_t i = 0; i < numVertices; i++) {
            if (!std::getline(f, line))
                throw std::runtime_error("Missing ASCII PLY vertex " +
                                         std::to_string(i) + ": " + path);
            std::istringstream iss(line);
            tokens.clear();
            std::string tok;
            while (iss >> tok) tokens.push_back(tok);
            if (tokens.size() < props.size())
                throw std::runtime_error("Incomplete ASCII PLY vertex " +
                                         std::to_string(i) + ": " + path);

            pts.xyz[i*3+0] = static_cast<float>(value(ix, i));
            pts.xyz[i*3+1] = static_cast<float>(value(iy, i));
            pts.xyz[i*3+2] = static_cast<float>(value(iz, i));

            if (hasColor) {
                pts.rgb[i*3+0] = colorByte(value(ir, i), props[ir].type);
                pts.rgb[i*3+1] = colorByte(value(ig, i), props[ig].type);
                pts.rgb[i*3+2] = colorByte(value(ib, i), props[ib].type);
            }
        }
    }

    return pts;
}
