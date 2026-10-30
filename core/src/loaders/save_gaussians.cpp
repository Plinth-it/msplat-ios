#include "loaders.hpp"
#include "msplat.hpp"
#include "atomic_output.hpp"
#include <fstream>
#include <algorithm>
#include <limits>
#include <numeric>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <zlib.h>

namespace fs = std::filesystem;

static const double C0 = 0.28209479177387814;
static constexpr float kSpzColorScale = 0.15f;
static constexpr float kSpzSqrtHalf = 0.7071067811865475f;
static constexpr uint32_t kSpzMagic = 0x5053474e; // "NGSP"
static constexpr uint32_t kSpzVersion = 3;
static constexpr int kSpzFractionalBits = 12;

struct SpzLegacyHeader {
    uint32_t magic = kSpzMagic;
    uint32_t version = kSpzVersion;
    uint32_t numPoints = 0;
    uint8_t shDegree = 0;
    uint8_t fractionalBits = kSpzFractionalBits;
    uint8_t flags = 0;
    uint8_t reserved = 0;
};
static_assert(sizeof(SpzLegacyHeader) == 16, "SPZ legacy header must be 16 bytes");

static float spzSigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static uint8_t byteClamp(float x) {
    return (uint8_t)std::clamp(std::round(x), 0.0f, 255.0f);
}

static uint8_t quantizeSh(float x, int bucketSize) {
    int q = (int)(std::round(x * 128.0f) + 128.0f);
    q = (q + bucketSize / 2) / bucketSize * bucketSize;
    return (uint8_t)std::clamp(q, 0, 255);
}

static int shDegreeForDim(int dim) {
    if (dim >= 24) return 4;
    if (dim >= 15) return 3;
    if (dim >= 8) return 2;
    if (dim >= 3) return 1;
    return 0;
}

static bool gzipBytes(const std::string &plain, std::vector<uint8_t> &out) {
    z_stream stream = {};
    if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK)
        return false;

    out.resize(std::max<size_t>(128, deflateBound(&stream, plain.size())));
    stream.next_in = (Bytef*)plain.data();
    stream.avail_in = (uInt)plain.size();

    int res;
    do {
        if (stream.total_out == out.size()) out.resize(out.size() * 2);
        stream.next_out = out.data() + stream.total_out;
        stream.avail_out = (uInt)(out.size() - stream.total_out);
        res = deflate(&stream, Z_FINISH);
    } while (res == Z_OK);

    bool ok = res == Z_STREAM_END;
    out.resize(stream.total_out);
    deflateEnd(&stream);
    return ok;
}

// SPZ stores a quaternion as its three smallest components at 10 bits each,
// plus 2 bits naming the one that was dropped — which is recovered from the
// unit-norm constraint, so the largest is the one worth dropping.
static void packSpzQuaternion(uint8_t out[4], float x, float y, float z, float w) {
    float norm = std::sqrt(x*x + y*y + z*z + w*w);
    if (norm <= 0.0f || !std::isfinite(norm)) {
        x = y = z = 0.0f; w = 1.0f;
    } else {
        x /= norm; y /= norm; z /= norm; w /= norm;
    }

    float q[4] = {x, y, z, w};
    int largest = 0;
    for (int i = 1; i < 4; i++)
        if (std::abs(q[i]) > std::abs(q[largest])) largest = i;

    bool negate = q[largest] < 0.0f;
    uint32_t packed = (uint32_t)largest;
    for (int i = 0; i < 4; i++) {
        if (i == largest) continue;
        uint32_t negbit = (q[i] < 0.0f) ^ negate;
        uint32_t mag = (uint32_t)(511.0f * (std::abs(q[i]) / kSpzSqrtHalf) + 0.5f);
        packed = (packed << 10u) | (negbit << 9u) | std::min<uint32_t>(mag, 511u);
    }

    out[0] = packed & 0xff;
    out[1] = (packed >> 8) & 0xff;
    out[2] = (packed >> 16) & 0xff;
    out[3] = (packed >> 24) & 0xff;
}

void saveGaussianPly(const std::string &path, GaussianParams &p, int step) {
    msplat_gpu_sync();

    msplat::detail::AtomicOutputFile output(path);
    std::ofstream o(output.temporary(), std::ios::binary | std::ios::trunc);
    if (!o.is_open()) throw std::runtime_error("Cannot open PLY file for writing: " + path);
    int64_t N = p.means.size(0);
    int numDc = (int)p.featuresDc.size(1);
    int frBases = (int)p.featuresRest.size(-2);
    int numFr = frBases * 3;

    o << "ply\nformat binary_little_endian 1.0\n";
    o << "comment msplat v" << step << "\n";
    o << "element vertex " << N << "\n";
    o << "property float x\nproperty float y\nproperty float z\n";
    o << "property float nx\nproperty float ny\nproperty float nz\n";
    for (int i = 0; i < numDc; i++) o << "property float f_dc_" << i << "\n";
    for (int i = 0; i < numFr; i++) o << "property float f_rest_" << i << "\n";
    o << "property float opacity\n";
    o << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n";
    o << "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n";
    o << "end_header\n";

    int floatsPerRow = 3 + 3 + numDc + numFr + 1 + 3 + 4;
    std::vector<float> row(floatsPerRow);
    const float *mp = p.means.data<float>(), *sp = p.scales.data<float>(), *qp = p.quats.data<float>();
    const float *dp = p.featuresDc.data<float>(), *op = p.opacities.data<float>();
    const float *frp = p.featuresRest.data<float>();

    for (int64_t i = 0; i < N; i++) {
        int c = 0;
        for (int j = 0; j < 3; j++)
            row[c++] = p.keepCrs ? (mp[i*3+j] / p.scale + p.translation[j]) : mp[i*3+j];
        row[c++] = 0; row[c++] = 0; row[c++] = 0; // normals
        for (int j = 0; j < numDc; j++) row[c++] = dp[i*numDc+j];
        // Transpose [frBases, 3] → [3, frBases] for PLY convention
        for (int ch = 0; ch < 3; ch++)
            for (int b = 0; b < frBases; b++)
                row[c++] = frp[i*frBases*3 + b*3 + ch];
        row[c++] = op[i];
        for (int j = 0; j < 3; j++)
            row[c++] = p.keepCrs ? sp[i*3+j] - std::log(p.scale) : sp[i*3+j];
        for (int j = 0; j < 4; j++) row[c++] = qp[i*4+j];

        o.write(reinterpret_cast<const char*>(row.data()), floatsPerRow * sizeof(float));
    }
    o.flush();
    if (!o) throw std::runtime_error("Failed while writing PLY file: " + path);
    o.close();
    if (!o) throw std::runtime_error("Failed to close PLY file: " + path);
    output.commit("PLY");
}

void saveGaussianSplat(const std::string &path, GaussianParams &p) {
    msplat_gpu_sync();

    msplat::detail::AtomicOutputFile output(path);
    std::ofstream o(output.temporary(), std::ios::binary | std::ios::trunc);
    if (!o.is_open()) throw std::runtime_error("Cannot open splat file for writing: " + path);
    int64_t N = p.means.size(0);
    const float *mp = p.means.data<float>(), *sp = p.scales.data<float>(), *qp = p.quats.data<float>();
    const float *dp = p.featuresDc.data<float>(), *op = p.opacities.data<float>();

    // Sort by size/opacity (largest first)
    std::vector<float> order(N);
    for (int64_t i = 0; i < N; i++) {
        float s = std::exp(sp[i*3]) + std::exp(sp[i*3+1]) + std::exp(sp[i*3+2]);
        if (p.keepCrs) s /= p.scale;
        order[i] = s / (1.0f + std::exp(-op[i]));
    }
    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return order[a] > order[b]; });

    for (int64_t ii = 0; ii < N; ii++) {
        size_t i = idx[ii];
        float m[3];
        for (int j = 0; j < 3; j++) m[j] = p.keepCrs ? (mp[i*3+j] / p.scale + p.translation[j]) : mp[i*3+j];
        o.write(reinterpret_cast<const char*>(m), 12);

        float sc[3];
        for (int j = 0; j < 3; j++) sc[j] = p.keepCrs ? (std::exp(sp[i*3+j]) / p.scale) : std::exp(sp[i*3+j]);
        o.write(reinterpret_cast<const char*>(sc), 12);

        uint8_t rgb[3];
        for (int j = 0; j < 3; j++) rgb[j] = (uint8_t)std::clamp(((double)dp[i*3+j] * C0 + 0.5) * 255.0, 0.0, 255.0);
        o.write(reinterpret_cast<const char*>(rgb), 3);

        float sig = 1.0f / (1.0f + std::exp(-op[i]));
        uint8_t a = (uint8_t)std::clamp(sig * 255.0f, 0.0f, 255.0f);
        o.write(reinterpret_cast<const char*>(&a), 1);

        uint8_t q[4];
        for (int j = 0; j < 4; j++) q[j] = (uint8_t)std::clamp(qp[i*4+j] * 128.0f + 128.0f, 0.0f, 255.0f);
        o.write(reinterpret_cast<const char*>(q), 4);
    }
    o.flush();
    if (!o) throw std::runtime_error("Failed while writing splat file: " + path);
    o.close();
    if (!o) throw std::runtime_error("Failed to close splat file: " + path);
    output.commit("splat");
}

void saveGaussianSpz(const std::string &path, GaussianParams &p) {
    msplat_gpu_sync();

    int64_t N = p.means.size(0);
    int frBases = (int)p.featuresRest.size(-2);
    int shDegree = shDegreeForDim(frBases);
    int shDim = shDegree == 0 ? 0 : (shDegree == 1 ? 3 : (shDegree == 2 ? 8 : (shDegree == 3 ? 15 : 24)));

    const float *mp = p.means.data<float>(), *sp = p.scales.data<float>(), *qp = p.quats.data<float>();
    const float *dp = p.featuresDc.data<float>(), *op = p.opacities.data<float>();
    const float *frp = p.featuresRest.data<float>();

    int32_t count = (int32_t)N;
    std::vector<uint8_t> positions((size_t)count * 9);
    std::vector<uint8_t> alphas(count);
    std::vector<uint8_t> colors((size_t)count * 3);
    std::vector<uint8_t> scales((size_t)count * 3);
    std::vector<uint8_t> rotations((size_t)count * 4);
    std::vector<uint8_t> sh((size_t)count * shDim * 3);

    const float posScale = (float)(1 << kSpzFractionalBits);
    for (int32_t outIdx = 0; outIdx < count; outIdx++) {
        size_t i = (size_t)outIdx;
        float pos[3];
        for (int j = 0; j < 3; j++)
            pos[j] = p.keepCrs ? (mp[i*3+j] / p.scale + p.translation[j]) : mp[i*3+j];

        for (int j = 0; j < 3; j++) {
            int32_t fixed = (int32_t)std::round(pos[j] * posScale);
            positions[((size_t)outIdx*3 + j) * 3 + 0] = fixed & 0xff;
            positions[((size_t)outIdx*3 + j) * 3 + 1] = (fixed >> 8) & 0xff;
            positions[((size_t)outIdx*3 + j) * 3 + 2] = (fixed >> 16) & 0xff;
        }

        alphas[outIdx] = byteClamp(spzSigmoid(op[i]) * 255.0f);
        for (int j = 0; j < 3; j++) {
            colors[(size_t)outIdx*3 + j] = byteClamp(dp[i*3+j] * (kSpzColorScale * 255.0f) + 127.5f);
            float s = p.keepCrs ? std::log(std::exp(sp[i*3+j]) / p.scale) : sp[i*3+j];
            scales[(size_t)outIdx*3 + j] = byteClamp((s + 10.0f) * 16.0f);
        }

        packSpzQuaternion(&rotations[(size_t)outIdx*4], qp[i*4+1], qp[i*4+2], qp[i*4+3], qp[i*4]);

        for (int b = 0; b < shDim; b++) {
            for (int ch = 0; ch < 3; ch++) {
                int bucket = b < 3 ? 8 : 16;
                sh[((size_t)outIdx * shDim + b) * 3 + ch] =
                    quantizeSh(frp[i*frBases*3 + b*3 + ch], bucket);
            }
        }
    }

    SpzLegacyHeader header;
    header.numPoints = (uint32_t)count;
    header.shDegree = (uint8_t)shDegree;

    std::ostringstream plain(std::ios::binary);
    plain.write(reinterpret_cast<const char*>(&header), sizeof(header));
    plain.write(reinterpret_cast<const char*>(positions.data()), positions.size());
    plain.write(reinterpret_cast<const char*>(alphas.data()), alphas.size());
    plain.write(reinterpret_cast<const char*>(colors.data()), colors.size());
    plain.write(reinterpret_cast<const char*>(scales.data()), scales.size());
    plain.write(reinterpret_cast<const char*>(rotations.data()), rotations.size());
    plain.write(reinterpret_cast<const char*>(sh.data()), sh.size());

    std::vector<uint8_t> compressed;
    std::string bytes = plain.str();
    if (!gzipBytes(bytes, compressed)) throw std::runtime_error("Failed to gzip SPZ data");

    msplat::detail::AtomicOutputFile output(path);
    std::ofstream o(output.temporary(), std::ios::binary | std::ios::trunc);
    if (!o.is_open()) throw std::runtime_error("Cannot open SPZ file for writing: " + path);
    o.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
    o.flush();
    if (!o) throw std::runtime_error("Failed while writing SPZ file: " + path);
    o.close();
    if (!o) throw std::runtime_error("Failed to close SPZ file: " + path);
    output.commit("SPZ");
}

namespace {

[[noreturn]] void malformedGaussianPly(const std::string &path, const std::string &reason) {
    throw std::runtime_error("Malformed Gaussian PLY file " + path + ": " + reason);
}

bool parseNonNegative(const std::string &token, uint64_t &out) {
    if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos ||
        token.size() > 19) return false;
    out = std::stoull(token);
    return true;
}

} // namespace

LoadedGaussians loadGaussianPly(const std::string &path, float scale, const float translation[3], bool keepCrs) {
    msplat_gpu_sync();

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open PLY file: " + path);

    // Parse header. Properties are located by name so files from other 3DGS
    // tools load when they use the same float layout, with or without normals.
    std::string line;
    uint64_t declaredPoints = 0;
    int step = 0;
    bool binaryLittleEndian = false, sawVertex = false, headerEnded = false;
    std::vector<std::string> props;

    auto stripLine = [](std::string &value) {
        while (!value.empty() &&
               (value.back() == '\r' || value.back() == ' ' || value.back() == '\t'))
            value.pop_back();
    };

    if (!std::getline(f, line)) throw std::runtime_error("Not a PLY file: " + path);
    stripLine(line);
    if (line != "ply") throw std::runtime_error("Not a PLY file: " + path);

    while (std::getline(f, line)) {
        stripLine(line);
        if (line == "end_header") {
            headerEnded = true;
            break;
        }
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "format") {
            std::string format;
            iss >> format;
            binaryLittleEndian = format == "binary_little_endian";
        } else if (token == "comment") {
            // saveGaussianPly writes "msplat v<step>"; older files used the
            // longer form.
            std::string rest;
            std::getline(iss >> std::ws, rest);
            for (const std::string prefix : {"msplat v", "Generated by msplat at iteration "}) {
                uint64_t parsed = 0;
                if (rest.rfind(prefix, 0) == 0 &&
                    parseNonNegative(rest.substr(prefix.size()), parsed) &&
                    parsed <= static_cast<uint64_t>(std::numeric_limits<int>::max()))
                    step = static_cast<int>(parsed);
            }
        } else if (token == "element") {
            std::string name, count;
            iss >> name >> count;
            if (name != "vertex" || sawVertex)
                malformedGaussianPly(path, "expected a single vertex element");
            if (!parseNonNegative(count, declaredPoints))
                malformedGaussianPly(path, "invalid vertex count");
            sawVertex = true;
        } else if (token == "property") {
            std::string type, name;
            iss >> type >> name;
            if (!sawVertex) malformedGaussianPly(path, "property declared before the vertex element");
            if (type != "float" && type != "float32")
                malformedGaussianPly(path, "property " + name + " must be float");
            if (std::find(props.begin(), props.end(), name) != props.end())
                malformedGaussianPly(path, "duplicate property " + name);
            props.push_back(name);
        }
    }

    if (!headerEnded) malformedGaussianPly(path, "missing end_header");
    if (!binaryLittleEndian) malformedGaussianPly(path, "format must be binary_little_endian");
    if (declaredPoints == 0) throw std::runtime_error("PLY has no vertices");
    if (declaredPoints > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        malformedGaussianPly(path, "vertex count is outside the supported range");

    auto offsetOf = [&](const std::string &name) -> int {
        const auto it = std::find(props.begin(), props.end(), name);
        if (it == props.end()) malformedGaussianPly(path, "missing property " + name);
        return static_cast<int>(it - props.begin());
    };
    auto countPrefixed = [&](const std::string &prefix) {
        int count = 0;
        while (std::find(props.begin(), props.end(), prefix + std::to_string(count)) != props.end())
            count++;
        return count;
    };
    const int numDc = countPrefixed("f_dc_");
    const int numFr = countPrefixed("f_rest_");
    if (numDc != 3) malformedGaussianPly(path, "expected f_dc_0...f_dc_2");
    if (numFr % 3 != 0) malformedGaussianPly(path, "f_rest count must be a multiple of 3");
    const int frBases = numFr / 3;
    if (frBases != 0 && frBases != 3 && frBases != 8 && frBases != 15 && frBases != 24)
        malformedGaussianPly(path, "f_rest count does not match an SH degree");

    const int xyz[3] = {offsetOf("x"), offsetOf("y"), offsetOf("z")};
    const int dc0 = offsetOf("f_dc_0");
    const int fr0 = numFr > 0 ? offsetOf("f_rest_0") : 0;
    const int opacity = offsetOf("opacity");
    const int scaleOffsets[3] = {offsetOf("scale_0"), offsetOf("scale_1"), offsetOf("scale_2")};
    const int rot[4] = {offsetOf("rot_0"), offsetOf("rot_1"), offsetOf("rot_2"), offsetOf("rot_3")};
    for (int i = 1; i < numDc; i++)
        if (offsetOf("f_dc_" + std::to_string(i)) != dc0 + i)
            malformedGaussianPly(path, "f_dc properties must be contiguous");
    for (int i = 1; i < numFr; i++)
        if (offsetOf("f_rest_" + std::to_string(i)) != fr0 + i)
            malformedGaussianPly(path, "f_rest properties must be contiguous");

    const size_t floatsPerRow = props.size();
    const std::streampos dataStart = f.tellg();
    f.seekg(0, std::ios::end);
    const std::streampos fileEnd = f.tellg();
    f.seekg(dataStart);
    if (dataStart < 0 || fileEnd < dataStart || !f)
        malformedGaussianPly(path, "cannot determine the payload size");
    const uint64_t payloadBytes = static_cast<uint64_t>(fileEnd - dataStart);
    if (declaredPoints > payloadBytes / (floatsPerRow * sizeof(float)))
        malformedGaussianPly(path, "vertex data is truncated");
    const size_t numPoints = static_cast<size_t>(declaredPoints);

    std::vector<float> meansRaw(numPoints * 3);
    std::vector<float> dcRaw(numPoints * numDc);
    std::vector<float> frRaw(numPoints * numFr);
    std::vector<float> opRaw(numPoints);
    std::vector<float> scRaw(numPoints * 3);
    std::vector<float> qtRaw(numPoints * 4);
    std::vector<float> row(floatsPerRow);

    for (size_t i = 0; i < numPoints; i++) {
        if (!f.read(reinterpret_cast<char*>(row.data()),
                    static_cast<std::streamsize>(floatsPerRow * sizeof(float))))
            malformedGaussianPly(path, "vertex data is truncated");
        for (int j = 0; j < 3; j++) meansRaw[i*3+j] = row[xyz[j]];
        std::copy_n(&row[dc0], numDc, &dcRaw[i*numDc]);
        if (numFr > 0) std::copy_n(&row[fr0], numFr, &frRaw[i*numFr]);
        opRaw[i] = row[opacity];
        for (int j = 0; j < 3; j++) scRaw[i*3+j] = row[scaleOffsets[j]];
        for (int j = 0; j < 4; j++) qtRaw[i*4+j] = row[rot[j]];
    }

    // CRS transform
    if (keepCrs) {
        for (size_t i = 0; i < numPoints; i++)
            for (int j = 0; j < 3; j++)
                meansRaw[i*3+j] = (meansRaw[i*3+j] - translation[j]) * scale;
        for (size_t i = 0; i < numPoints * 3; i++)
            scRaw[i] += std::log(scale);
    }

    // Upload to GPU
    LoadedGaussians g;
    g.step = step;
    auto upload = [](std::vector<int64_t> shape, const float *src, size_t bytes) {
        MTensor t = gpu_empty(shape, DType::Float32);
        memcpy(t.data_ptr(), src, bytes);
        return t;
    };
    g.means = upload({(int64_t)numPoints, 3}, meansRaw.data(), meansRaw.size() * 4);
    g.featuresDc = upload({(int64_t)numPoints, (int64_t)numDc}, dcRaw.data(), dcRaw.size() * 4);
    g.opacities = upload({(int64_t)numPoints, 1}, opRaw.data(), opRaw.size() * 4);
    g.scales = upload({(int64_t)numPoints, 3}, scRaw.data(), scRaw.size() * 4);
    g.quats = upload({(int64_t)numPoints, 4}, qtRaw.data(), qtRaw.size() * 4);

    // Transpose featuresRest: PLY [N, 3, frBases] → internal [N, frBases, 3]
    g.featuresRest = gpu_empty({(int64_t)numPoints, (int64_t)frBases, 3}, DType::Float32);
    float *frOut = g.featuresRest.data<float>();
    for (size_t i = 0; i < numPoints; i++)
        for (int ch = 0; ch < 3; ch++)
            for (int b = 0; b < frBases; b++)
                frOut[i*frBases*3 + b*3 + ch] = frRaw[i*numFr + ch*frBases + b];

    // numPoints and step available via returned struct
    return g;
}
