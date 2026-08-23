#include "loaders.hpp"
#include "msplat.hpp"
#include "atomic_output.hpp"
#include <fstream>
#include <algorithm>
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
            row[c++] = p.keepCrs ? std::log(std::exp(sp[i*3+j]) / p.scale) : sp[i*3+j];
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

LoadedGaussians loadGaussianPly(const std::string &path, float scale, const float translation[3], bool keepCrs) {
    msplat_gpu_sync();

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open PLY file: " + path);

    // Parse header
    std::string line;
    int numPoints = 0, step = 0;
    int numDc = 0, numFr = 0;

    std::getline(f, line); // "ply"
    if (line.find("ply") == std::string::npos) throw std::runtime_error("Not a PLY file: " + path);
    std::getline(f, line); // "format binary_little_endian 1.0"

    while (std::getline(f, line)) {
        if (line == "end_header") break;

        const std::string iterPrefix = "comment Generated by msplat at iteration ";
        if (line.rfind(iterPrefix, 0) == 0)
            step = std::stoi(line.substr(iterPrefix.length()));

        const std::string vertexPrefix = "element vertex ";
        if (line.rfind(vertexPrefix, 0) == 0)
            numPoints = std::stoi(line.substr(vertexPrefix.length()));

        if (line.rfind("property float f_dc_", 0) == 0) numDc++;
        if (line.rfind("property float f_rest_", 0) == 0) numFr++;
    }

    if (numPoints == 0) throw std::runtime_error("PLY has no vertices");
    int frBases = numFr / 3;

    // Read binary data: xyz(3) + normals(3) + f_dc(numDc) + f_rest(numFr) + opacity(1) + scale(3) + rot(4)
    std::vector<float> meansRaw(numPoints * 3);
    std::vector<float> dcRaw(numPoints * numDc);
    std::vector<float> frRaw(numPoints * numFr);
    std::vector<float> opRaw(numPoints);
    std::vector<float> scRaw(numPoints * 3);
    std::vector<float> qtRaw(numPoints * 4);
    float normals[3];

    for (int i = 0; i < numPoints; i++) {
        f.read(reinterpret_cast<char*>(&meansRaw[i*3]), 12);
        f.read(reinterpret_cast<char*>(normals), 12);
        f.read(reinterpret_cast<char*>(&dcRaw[i*numDc]), numDc * 4);
        f.read(reinterpret_cast<char*>(&frRaw[i*numFr]), numFr * 4);
        f.read(reinterpret_cast<char*>(&opRaw[i]), 4);
        f.read(reinterpret_cast<char*>(&scRaw[i*3]), 12);
        f.read(reinterpret_cast<char*>(&qtRaw[i*4]), 16);
    }

    // CRS transform
    if (keepCrs) {
        for (int i = 0; i < numPoints; i++)
            for (int j = 0; j < 3; j++)
                meansRaw[i*3+j] = (meansRaw[i*3+j] - translation[j]) * scale;
        for (int i = 0; i < numPoints * 3; i++)
            scRaw[i] = std::log(scale * std::exp(scRaw[i]));
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
    for (int i = 0; i < numPoints; i++)
        for (int ch = 0; ch < 3; ch++)
            for (int b = 0; b < frBases; b++)
                frOut[i*frBases*3 + b*3 + ch] = frRaw[i*numFr + ch*frBases + b];

    // numPoints and step available via returned struct
    return g;
}
