#include "loaders.hpp"
#include "dataset_errors.hpp"

#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

template <typename T>
class CFHandle {
public:
    explicit CFHandle(T value = nullptr) : value_(value) {}
    ~CFHandle() {
        if (value_) CFRelease(value_);
    }

    CFHandle(const CFHandle &) = delete;
    CFHandle &operator=(const CFHandle &) = delete;

    T get() const { return value_; }
    explicit operator bool() const { return value_ != nullptr; }

private:
    T value_;
};

[[noreturn]] void fail(const char *expression, int line,
                       const std::string &detail = {}) {
    std::string message = "line " + std::to_string(line) + ": " + expression;
    if (!detail.empty()) message += " (" + detail + ")";
    throw std::runtime_error(message);
}

#define CHECK(condition) \
    do { if (!(condition)) fail(#condition, __LINE__); } while (false)

struct RGB8 {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
};

constexpr std::array<RGB8, 6> gridColors = {{
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {255, 255, 0},
    {255, 0, 255},
    {0, 255, 255},
}};

std::vector<uint8_t> rgbaGrid(int width, int height,
                              const std::vector<RGB8> &colors) {
    CHECK(static_cast<size_t>(width * height) == colors.size());
    std::vector<uint8_t> rgba(colors.size() * 4);
    for (size_t index = 0; index < colors.size(); ++index) {
        rgba[index * 4 + 0] = colors[index].red;
        rgba[index * 4 + 1] = colors[index].green;
        rgba[index * 4 + 2] = colors[index].blue;
        rgba[index * 4 + 3] = 255;
    }
    return rgba;
}

std::vector<uint8_t> solidRGBA(int width, int height, RGB8 color) {
    std::vector<uint8_t> rgba(static_cast<size_t>(width) * height * 4);
    for (size_t index = 0; index < rgba.size() / 4; ++index) {
        rgba[index * 4 + 0] = color.red;
        rgba[index * 4 + 1] = color.green;
        rgba[index * 4 + 2] = color.blue;
        rgba[index * 4 + 3] = 255;
    }
    return rgba;
}

void writeTIFF(const fs::path &path, int width, int height,
               const std::vector<uint8_t> &rgba, int orientation) {
    CHECK(width > 0 && height > 0);
    CHECK(rgba.size() == static_cast<size_t>(width) * height * 4);
    CHECK(orientation >= 1 && orientation <= 8);

    CFHandle<CGDataProviderRef> provider(CGDataProviderCreateWithData(
        nullptr, rgba.data(), rgba.size(), nullptr));
    CHECK(provider);
    CFHandle<CGColorSpaceRef> colorSpace(
        CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
    CHECK(colorSpace);

    const CGBitmapInfo bitmapInfo = static_cast<CGBitmapInfo>(
        kCGBitmapByteOrder32Big | kCGImageAlphaPremultipliedLast);
    CFHandle<CGImageRef> image(CGImageCreate(
        static_cast<size_t>(width), static_cast<size_t>(height), 8, 32,
        static_cast<size_t>(width) * 4, colorSpace.get(), bitmapInfo,
        provider.get(), nullptr, false, kCGRenderingIntentDefault));
    CHECK(image);

    CFHandle<CFURLRef> url(CFURLCreateFromFileSystemRepresentation(
        nullptr, reinterpret_cast<const UInt8 *>(path.c_str()),
        static_cast<CFIndex>(path.string().size()), false));
    CHECK(url);
    CFHandle<CGImageDestinationRef> destination(
        CGImageDestinationCreateWithURL(url.get(), CFSTR("public.tiff"), 1,
                                        nullptr));
    CHECK(destination);

    int32_t orientationValue = orientation;
    CFHandle<CFNumberRef> orientationNumber(CFNumberCreate(
        nullptr, kCFNumberSInt32Type, &orientationValue));
    CHECK(orientationNumber);
    const void *keys[] = {kCGImagePropertyOrientation};
    const void *values[] = {orientationNumber.get()};
    CFHandle<CFDictionaryRef> properties(CFDictionaryCreate(
        nullptr, keys, values, 1,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks));
    CHECK(properties);

    CGImageDestinationAddImage(destination.get(), image.get(), properties.get());
    CHECK(CGImageDestinationFinalize(destination.get()));
}

RGB8 pixel8(const Image &image, int x, int y) {
    CHECK(x >= 0 && x < image.width);
    CHECK(y >= 0 && y < image.height);
    const size_t index = (static_cast<size_t>(y) * image.width + x) * 3;
    auto quantize = [](float value) {
        return static_cast<uint8_t>(std::lround(value * 255.0f));
    };
    return {
        quantize(image.data[index + 0]),
        quantize(image.data[index + 1]),
        quantize(image.data[index + 2]),
    };
}

bool near(RGB8 lhs, RGB8 rhs, int tolerance = 2) {
    return std::abs(static_cast<int>(lhs.red) - rhs.red) <= tolerance &&
           std::abs(static_cast<int>(lhs.green) - rhs.green) <= tolerance &&
           std::abs(static_cast<int>(lhs.blue) - rhs.blue) <= tolerance;
}

void checkPixel(const Image &image, int x, int y, RGB8 expected) {
    const RGB8 actual = pixel8(image, x, y);
    if (!near(actual, expected)) {
        fail("pixel color", __LINE__,
             "at " + std::to_string(x) + "," + std::to_string(y) +
             " expected " + std::to_string(expected.red) + "," +
             std::to_string(expected.green) + "," +
             std::to_string(expected.blue) + " got " +
             std::to_string(actual.red) + "," +
             std::to_string(actual.green) + "," +
             std::to_string(actual.blue));
    }
}

uint8_t coverageAt(const CoverageMask &mask, int x, int y) {
    CHECK(x >= 0 && x < mask.width);
    CHECK(y >= 0 && y < mask.height);
    return mask.data[static_cast<size_t>(y) * mask.width + x];
}

void checkCoverageNear(const CoverageMask &mask, int x, int y,
                       int expected, int tolerance = 2) {
    const int actual = coverageAt(mask, x, y);
    if (std::abs(actual - expected) > tolerance) {
        fail("mask coverage", __LINE__,
             "at " + std::to_string(x) + "," + std::to_string(y) +
             " expected " + std::to_string(expected) + " got " +
             std::to_string(actual));
    }
}

void checkSameImage(const Image &lhs, const Image &rhs) {
    CHECK(lhs.width == rhs.width);
    CHECK(lhs.height == rhs.height);
    CHECK(lhs.data.size() == rhs.data.size());
    for (int y = 0; y < lhs.height; ++y) {
        for (int x = 0; x < lhs.width; ++x) {
            const RGB8 left = pixel8(lhs, x, y);
            const RGB8 right = pixel8(rhs, x, y);
            if (!near(left, right)) {
                fail("raw EXIF decode preserves pixel positions", __LINE__,
                     "at " + std::to_string(x) + "," + std::to_string(y));
            }
        }
    }
}

template <typename Exception, typename Operation>
void checkThrows(Operation operation, const std::string &messageFragment) {
    try {
        operation();
    } catch (const Exception &error) {
        if (!messageFragment.empty()) {
            CHECK(std::string(error.what()).find(messageFragment) !=
                  std::string::npos);
        }
        return;
    }
    fail("expected exception", __LINE__, messageFragment);
}

template <typename Operation>
void checkStage(const char *name, Operation operation) {
    try {
        operation();
    } catch (const std::exception &error) {
        throw std::runtime_error(std::string(name) + ": " + error.what());
    }
}

struct TempDirectory {
    fs::path path;

    TempDirectory() {
        std::string pattern =
            (fs::temp_directory_path() / "msplat-image-io-test-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        const char *created = mkdtemp(writable.data());
        if (!created) {
            throw std::runtime_error("could not create temporary test directory");
        }
        path = created;
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void checkPNGWriteReadRoundTrip(const TempDirectory &temporary) {
    Image source;
    source.width = 3;
    source.height = 2;
    source.data.resize(gridColors.size() * 3);
    for (size_t index = 0; index < gridColors.size(); ++index) {
        source.data[index * 3 + 0] = gridColors[index].red / 255.0f;
        source.data[index * 3 + 1] = gridColors[index].green / 255.0f;
        source.data[index * 3 + 2] = gridColors[index].blue / 255.0f;
    }

    const fs::path path = temporary.path / "write-read-round-trip.png";
    imwriteRGB(path.string(), source);
    const ImageSourceInfo info = inspectImageSource(path.string());
    CHECK(info.rawWidth == source.width);
    CHECK(info.rawHeight == source.height);
    const Image decoded = imreadRGB(
        path.string(), info, source.width, source.height, false);
    for (int index = 0; index < 6; ++index) {
        checkPixel(decoded, index % 3, index / 3, gridColors[index]);
    }
}

void checkIndependentPPMRowOrder(const TempDirectory &temporary) {
    const fs::path path = temporary.path / "top-left-order.ppm";
    std::array<uint8_t, gridColors.size() * 3> rgb{};
    for (size_t index = 0; index < gridColors.size(); ++index) {
        rgb[index * 3 + 0] = gridColors[index].red;
        rgb[index * 3 + 1] = gridColors[index].green;
        rgb[index * 3 + 2] = gridColors[index].blue;
    }

    // PPM P6 defines raster bytes from the top-left, row by row. This fixture
    // bypasses Core Graphics encoding so the decode row-order assertion is
    // independent of Quartz coordinate conventions.
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream << "P6\n3 2\n255\n";
        stream.write(reinterpret_cast<const char *>(rgb.data()),
                     static_cast<std::streamsize>(rgb.size()));
        CHECK(static_cast<bool>(stream));
    }

    const ImageSourceInfo info = inspectImageSource(path.string());
    CHECK(info.rawWidth == 3);
    CHECK(info.rawHeight == 2);
    CHECK(info.exifOrientation == 1);
    const Image decoded = imreadRGB(path.string(), info, 3, 2, false);
    CHECK(decoded.width == 3);
    CHECK(decoded.height == 2);
    for (int index = 0; index < 6; ++index) {
        checkPixel(decoded, index % 3, index / 3, gridColors[index]);
    }
}

void checkMetadataAndRawOrientation(const TempDirectory &temporary) {
    std::vector<RGB8> colors(gridColors.begin(), gridColors.end());
    const std::vector<uint8_t> rgba = rgbaGrid(3, 2, colors);
    constexpr std::array<std::array<int, 6>, 8> orientedPixelIndices = {{
        {{0, 1, 2, 3, 4, 5}},
        {{2, 1, 0, 5, 4, 3}},
        {{5, 4, 3, 2, 1, 0}},
        {{3, 4, 5, 0, 1, 2}},
        {{0, 3, 1, 4, 2, 5}},
        {{3, 0, 4, 1, 5, 2}},
        {{5, 2, 4, 1, 3, 0}},
        {{2, 5, 1, 4, 0, 3}},
    }};
    Image rawBaseline;

    for (int orientation = 1; orientation <= 8; ++orientation) {
        const fs::path path = temporary.path /
            ("orientation-" + std::to_string(orientation) + ".tiff");
        writeTIFF(path, 3, 2, rgba, orientation);

        const ImageSourceInfo info = inspectImageSource(path.string());
        CHECK(info.rawWidth == 3);
        CHECK(info.rawHeight == 2);
        CHECK(info.exifOrientation == orientation);
        CHECK(info.orientedWidth == (orientation >= 5 ? 2 : 3));
        CHECK(info.orientedHeight == (orientation >= 5 ? 3 : 2));

        const Image decoded = imreadRGB(path.string(), info, 3, 2, false);
        CHECK(decoded.width == 3);
        CHECK(decoded.height == 2);
        if (orientation == 1) {
            rawBaseline = decoded;
        } else {
            checkSameImage(rawBaseline, decoded);
        }

        const int orientedWidth = orientation >= 5 ? 2 : 3;
        const int orientedHeight = orientation >= 5 ? 3 : 2;
        const Image oriented = imreadRGB(
            path.string(), info, orientedWidth, orientedHeight, true);
        CHECK(oriented.width == orientedWidth);
        CHECK(oriented.height == orientedHeight);
        for (int index = 0; index < 6; ++index) {
            checkPixel(
                oriented, index % orientedWidth, index / orientedWidth,
                gridColors[orientedPixelIndices[
                    static_cast<size_t>(orientation - 1)][index]]);
        }
    }

    // Lock the byte/channel interpretation as well as cross-orientation
    // equality.
    checkPixel(rawBaseline, 0, 0, gridColors[0]);
    checkPixel(rawBaseline, 1, 0, gridColors[1]);
    checkPixel(rawBaseline, 2, 0, gridColors[2]);
    checkPixel(rawBaseline, 0, 1, gridColors[3]);
    checkPixel(rawBaseline, 1, 1, gridColors[4]);
    checkPixel(rawBaseline, 2, 1, gridColors[5]);
}

void checkExactTargetSizeAndColor(const TempDirectory &temporary) {
    constexpr RGB8 color{32, 96, 224};
    const fs::path path = temporary.path / "odd-source.tiff";
    writeTIFF(path, 13, 9, solidRGBA(13, 9, color), 1);
    const ImageSourceInfo info = inspectImageSource(path.string());

    const Image decoded = imreadRGB(path.string(), info, 5, 3, false);
    CHECK(decoded.width == 5);
    CHECK(decoded.height == 3);
    CHECK(decoded.data.size() == 5 * 3 * 3);
    for (int y = 0; y < decoded.height; ++y) {
        for (int x = 0; x < decoded.width; ++x) {
            checkPixel(decoded, x, y, color);
        }
    }

    checkThrows<std::runtime_error>(
        [&] { (void)imreadRGB(path.string(), info, 0, 3, false); },
        "invalid dimensions");
    checkThrows<std::runtime_error>(
        [&] { (void)imreadRGB(path.string(), info, 5, 0, false); },
        "invalid dimensions");
    checkThrows<std::runtime_error>(
        [&] { (void)imreadRGB(path.string(), info, -1, 3, false); },
        "invalid dimensions");

    ImageSourceInfo stale = info;
    stale.exifOrientation = 2;
    checkThrows<std::runtime_error>(
        [&] { (void)imreadRGB(path.string(), stale, 5, 3, false); },
        "changed while loading");
}

void checkCoverageMaskDecodeAndResize(const TempDirectory &temporary) {
    const fs::path luminancePath = temporary.path / "luminance-mask.tiff";
    const std::vector<RGB8> colors = {
        {255, 0, 0}, {0, 255, 0},
        {0, 0, 255}, {255, 255, 255},
    };
    writeTIFF(luminancePath, 2, 2, rgbaGrid(2, 2, colors), 1);
    const ImageSourceInfo luminanceInfo =
        inspectImageSource(luminancePath.string());
    const CoverageMask luminance = imreadCoverageMask(
        luminancePath.string(), luminanceInfo, 2, 2, false,
        TrainingMaskChannel::Luminance);
    CHECK(luminance.width == 2);
    CHECK(luminance.height == 2);
    checkCoverageNear(luminance, 0, 0, 54);
    checkCoverageNear(luminance, 1, 0, 182);
    checkCoverageNear(luminance, 0, 1, 18);
    checkCoverageNear(luminance, 1, 1, 255);

    const fs::path alphaPath = temporary.path / "alpha-mask.tiff";
    std::vector<uint8_t> alphaRGBA(2 * 2 * 4, 0);
    const std::array<uint8_t, 4> alphaValues = {0, 64, 128, 255};
    for (size_t index = 0; index < alphaValues.size(); ++index)
        alphaRGBA[index * 4 + 3] = alphaValues[index];
    writeTIFF(alphaPath, 2, 2, alphaRGBA, 1);
    const ImageSourceInfo alphaInfo = inspectImageSource(alphaPath.string());
    const CoverageMask alpha = imreadCoverageMask(
        alphaPath.string(), alphaInfo, 2, 2, false,
        TrainingMaskChannel::Alpha);
    for (int index = 0; index < 4; ++index) {
        checkCoverageNear(alpha, index % 2, index / 2,
                          alphaValues[static_cast<size_t>(index)], 0);
    }

    const fs::path premultipliedPath =
        temporary.path / "premultiplied-luminance-mask.tiff";
    writeTIFF(premultipliedPath, 1, 1, {128, 0, 0, 128}, 1);
    const ImageSourceInfo premultipliedInfo =
        inspectImageSource(premultipliedPath.string());
    const CoverageMask premultiplied = imreadCoverageMask(
        premultipliedPath.string(), premultipliedInfo, 1, 1, false,
        TrainingMaskChannel::Luminance);
    checkCoverageNear(premultiplied, 0, 0, 27);

    CoverageMask soft;
    soft.width = 2;
    soft.height = 2;
    soft.data = {0, 64, 128, 255};
    const CoverageMask averaged = resizeCoverageArea(soft, 1, 1);
    CHECK(averaged.width == 1);
    CHECK(averaged.height == 1);
    CHECK(averaged.data == std::vector<uint8_t>({112}));

    const fs::path areaPath = temporary.path / "area-alpha-mask.tiff";
    const std::array<uint8_t, 16> areaValues = {
        255, 0,   0,   255,
        0,   0,   255, 255,
        128, 128, 64,  64,
        128, 128, 64,  64,
    };
    std::vector<uint8_t> areaRGBA(4 * 4 * 4, 0);
    for (size_t index = 0; index < areaValues.size(); ++index) {
        areaRGBA[index * 4 + 3] = areaValues[index];
    }
    writeTIFF(areaPath, 4, 4, areaRGBA, 1);
    const ImageSourceInfo areaInfo = inspectImageSource(areaPath.string());
    const CoverageMask decodedArea = imreadCoverageMask(
        areaPath.string(), areaInfo, 2, 2, false,
        TrainingMaskChannel::Alpha);
    CHECK(decodedArea.width == 2);
    CHECK(decodedArea.height == 2);
    CHECK(decodedArea.data == std::vector<uint8_t>({64, 191, 128, 64}));

    const fs::path noAlphaPath = temporary.path / "no-alpha.ppm";
    {
        std::ofstream stream(noAlphaPath, std::ios::binary | std::ios::trunc);
        const std::array<uint8_t, 6> pixels = {255, 255, 255, 0, 0, 0};
        stream << "P6\n2 1\n255\n";
        stream.write(reinterpret_cast<const char *>(pixels.data()),
                     static_cast<std::streamsize>(pixels.size()));
        CHECK(static_cast<bool>(stream));
    }
    const ImageSourceInfo noAlphaInfo = inspectImageSource(noAlphaPath.string());
    checkThrows<msplat::InvalidDatasetError>(
        [&] {
            (void)imreadCoverageMask(
                noAlphaPath.string(), noAlphaInfo, 2, 1, false,
                TrainingMaskChannel::Alpha);
        },
        "no alpha channel");
}

void checkCoverageMaskOrientation(const TempDirectory &temporary) {
    std::vector<uint8_t> rgba(3 * 2 * 4, 0);
    const std::array<uint8_t, 6> values = {16, 32, 48, 64, 80, 96};
    for (size_t index = 0; index < values.size(); ++index)
        rgba[index * 4 + 3] = values[index];
    constexpr std::array<std::array<int, 6>, 8> orientedPixelIndices = {{
        {{0, 1, 2, 3, 4, 5}},
        {{2, 1, 0, 5, 4, 3}},
        {{5, 4, 3, 2, 1, 0}},
        {{3, 4, 5, 0, 1, 2}},
        {{0, 3, 1, 4, 2, 5}},
        {{3, 0, 4, 1, 5, 2}},
        {{5, 2, 4, 1, 3, 0}},
        {{2, 5, 1, 4, 0, 3}},
    }};

    for (int orientation = 1; orientation <= 8; ++orientation) {
        const fs::path path = temporary.path /
            ("oriented-alpha-mask-" + std::to_string(orientation) + ".tiff");
        writeTIFF(path, 3, 2, rgba, orientation);

        const ImageSourceInfo info = inspectImageSource(path.string());
        const int orientedWidth = orientation >= 5 ? 2 : 3;
        const int orientedHeight = orientation >= 5 ? 3 : 2;
        const CoverageMask oriented = imreadCoverageMask(
            path.string(), info, orientedWidth, orientedHeight, true,
            TrainingMaskChannel::Alpha);
        CHECK(oriented.width == orientedWidth);
        CHECK(oriented.height == orientedHeight);
        for (int index = 0; index < 6; ++index) {
            checkCoverageNear(
                oriented, index % orientedWidth, index / orientedWidth,
                values[orientedPixelIndices[
                    static_cast<size_t>(orientation - 1)][index]], 0);
        }
    }
}

void checkMirroredCameraOrientationRejected(
    const TempDirectory &temporary) {
    for (int orientation : {2, 4, 5, 7}) {
        const fs::path path = temporary.path /
            ("mirrored-camera-" + std::to_string(orientation) + ".tiff");
        writeTIFF(path, 3, 2,
                  solidRGBA(3, 2, {24, 144, 208}), orientation);

        Camera camera;
        camera.filePath = path.string();
        camera.rasterOrientation = RasterOrientation::ExifNormalized;
        camera.width = orientation >= 5 ? 2 : 3;
        camera.height = orientation >= 5 ? 3 : 2;
        camera.fx = 10.0f;
        camera.fy = 10.0f;
        camera.cx = static_cast<float>(camera.width) * 0.5f;
        camera.cy = static_cast<float>(camera.height) * 0.5f;
        checkThrows<msplat::InvalidDatasetError>(
            [&] { camera.loadImage(1.0f); },
            "cannot represent mirrored EXIF orientation");
    }
}

Image coordinateImage(int width, int height) {
    Image image;
    image.width = width;
    image.height = height;
    image.data.resize(static_cast<size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t pixel = static_cast<size_t>(y) * width + x;
            image.data[pixel * 3 + 0] =
                static_cast<float>(pixel * 3 + 0) / 255.0f;
            image.data[pixel * 3 + 1] =
                static_cast<float>(pixel * 3 + 1) / 255.0f;
            image.data[pixel * 3 + 2] =
                static_cast<float>(pixel * 3 + 2) / 255.0f;
        }
    }
    return image;
}

CoverageMask coordinateMask(int width, int height) {
    CoverageMask mask;
    mask.width = width;
    mask.height = height;
    mask.data.resize(static_cast<size_t>(width) * height);
    for (size_t index = 0; index < mask.data.size(); ++index)
        mask.data[index] = static_cast<uint8_t>(index + 1);
    return mask;
}

void checkIdentityUndistortion() {
    const Image image = coordinateImage(5, 4);
    const CoverageMask mask = coordinateMask(5, 4);
    const auto result = undistortImageAndCoverageMask(
        image, mask, 8.0f, 7.0f, 2.5f, 2.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    CHECK(result.width == 5);
    CHECK(result.height == 4);
    CHECK(result.cx == 2.5f);
    CHECK(result.cy == 2.0f);
    CHECK(result.image.data == image.data);
    CHECK(result.coverageMask.data == mask.data);
}

void checkUndistortionRasterEdges() {
    const Image image = coordinateImage(9, 3);

    // Every output center remains inside the distorted raster. Sampling the
    // center locus instead of the raster edges would incorrectly crop a column
    // from both sides.
    const auto weak = undistortImage(
        image, 10.0f, 1000.0f, 4.5f, 1.5f,
        0.3f, 0.0f, 0.0f, 0.0f, 0.0f);
    CHECK(weak.width == 9);
    CHECK(weak.height == 3);
    CHECK(weak.cx == 4.5f);
    CHECK(weak.cy == 1.5f);

    const auto strong = undistortImage(
        image, 10.0f, 1000.0f, 4.5f, 1.5f,
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    CHECK(strong.width == 7);
    CHECK(strong.height == 3);
    CHECK(strong.cx == 3.5f);
    CHECK(strong.cy == 1.5f);
}

void checkUndistortionPixelCenter() {
    const Image image = coordinateImage(9, 9);
    const CoverageMask mask = coordinateMask(9, 9);
    const auto result = undistortImageAndCoverageMask(
        image, mask, 10.0f, 10.0f, 4.5f, 4.5f,
        0.0f, 0.0f, 0.0f, 0.10f, 0.0f);

    const int principalX = static_cast<int>(std::lround(result.cx - 0.5f));
    const int principalY = static_cast<int>(std::lround(result.cy - 0.5f));
    CHECK(principalX >= 0 && principalX < result.width);
    CHECK(principalY >= 0 && principalY < result.height);

    const size_t sourcePixel = 4 * 9 + 4;
    const size_t outputPixel =
        static_cast<size_t>(principalY) * result.width + principalX;
    for (int channel = 0; channel < 3; ++channel) {
        CHECK(result.image.data[outputPixel * 3 + channel] ==
              image.data[sourcePixel * 3 + channel]);
    }
    CHECK(result.coverageMask.data[outputPixel] == mask.data[sourcePixel]);
}

void checkExtremeDistortionRejected() {
    const Image image = coordinateImage(2, 2);
    checkThrows<msplat::InvalidDatasetError>(
        [&] {
            (void)undistortImage(
                image, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0e20f, 0.0f, 0.0f, 0.0f, 0.0f);
        },
        "Distortion model");
}

void checkJointUndistortion() {
    Image image;
    image.width = 13;
    image.height = 9;
    image.data.resize(13 * 9 * 3);
    CoverageMask mask;
    mask.width = 13;
    mask.height = 9;
    mask.data.resize(13 * 9);
    for (int y = 0; y < 9; ++y) {
        for (int x = 0; x < 13; ++x) {
            const size_t pixel = static_cast<size_t>(y) * 13 + x;
            const uint8_t coverage = static_cast<uint8_t>(
                16 + ((x * 17 + y * 29) % 224));
            mask.data[pixel] = coverage;
            for (int channel = 0; channel < 3; ++channel) {
                image.data[pixel * 3 + channel] = coverage / 255.0f;
            }
        }
    }

    const auto result = undistortImageAndCoverageMask(
        image, mask, 10.0f, 10.0f, 6.0f, 4.0f,
        0.08f, -0.01f, 0.002f, -0.003f, 0.0f);
    CHECK(result.image.width == result.width);
    CHECK(result.image.height == result.height);
    CHECK(result.coverageMask.width == result.width);
    CHECK(result.coverageMask.height == result.height);
    for (size_t pixel = 0; pixel < result.coverageMask.data.size(); ++pixel) {
        const int imageByte = static_cast<int>(std::lround(
            result.image.data[pixel * 3] * 255.0f));
        CHECK(std::abs(imageByte - result.coverageMask.data[pixel]) <= 1);
    }
}

void checkMaskedCameraLoading(const TempDirectory &temporary) {
    const fs::path imagePath = temporary.path / "masked-camera-rgb.tiff";
    writeTIFF(imagePath, 13, 9,
              solidRGBA(13, 9, {24, 144, 208}), 1);

    const fs::path maskPath = temporary.path / "masked-camera-mask.tiff";
    std::vector<uint8_t> maskRGBA(13 * 9 * 4, 0);
    for (size_t index = 0; index < maskRGBA.size() / 4; ++index)
        maskRGBA[index * 4 + 3] = 128;
    writeTIFF(maskPath, 13, 9, maskRGBA, 1);

    Camera camera;
    camera.filePath = imagePath.string();
    camera.trainingMask = TrainingMaskDescriptor{
        maskPath.string(), TrainingMaskChannel::Alpha};
    camera.width = 13;
    camera.height = 9;
    camera.fx = 13.0f;
    camera.fy = 9.0f;
    camera.loadImage(2.5f);
    CHECK(camera.image.width == 5);
    CHECK(camera.image.height == 3);
    CHECK(camera.coverageMask.width == 5);
    CHECK(camera.coverageMask.height == 3);
    for (uint8_t value : camera.coverageMask.data) CHECK(value == 128);

    const CoverageMask &coarse = camera.getCoverageMask(2);
    CHECK(coarse.width == 2);
    CHECK(coarse.height == 1);
    for (uint8_t value : coarse.data) CHECK(value == 128);

    const fs::path mismatchedPath = temporary.path / "wrong-size-mask.tiff";
    writeTIFF(mismatchedPath, 12, 9,
              solidRGBA(12, 9, {255, 255, 255}), 1);
    Camera mismatched;
    mismatched.filePath = imagePath.string();
    mismatched.trainingMask = TrainingMaskDescriptor{
        mismatchedPath.string(), TrainingMaskChannel::Luminance};
    mismatched.width = 13;
    mismatched.height = 9;
    checkThrows<msplat::InvalidDatasetError>(
        [&] { mismatched.loadImage(1.0f); },
        "do not match");

    const fs::path zeroPath = temporary.path / "zero-mask.tiff";
    writeTIFF(zeroPath, 13, 9, solidRGBA(13, 9, {0, 0, 0}), 1);
    Camera zero;
    zero.filePath = imagePath.string();
    zero.trainingMask = TrainingMaskDescriptor{
        zeroPath.string(), TrainingMaskChannel::Luminance};
    zero.width = 13;
    zero.height = 9;
    checkThrows<msplat::InvalidDatasetError>(
        [&] { zero.loadImage(1.0f); },
        "zero coverage");
}

void checkFractionalCameraScale(const TempDirectory &temporary) {
    constexpr RGB8 color{24, 144, 208};
    const fs::path path = temporary.path / "camera-orientation-6.tiff";
    writeTIFF(path, 13, 9, solidRGBA(13, 9, color), 6);

    Camera camera;
    camera.filePath = path.string();
    camera.width = 13;
    camera.height = 9;
    camera.fx = 130.0f;
    camera.fy = 90.0f;
    camera.cx = 6.0f;
    camera.cy = 4.0f;

    camera.loadImage(2.5f);
    CHECK(camera.width == 5);
    CHECK(camera.height == 3);
    CHECK(camera.image.width == 5);
    CHECK(camera.image.height == 3);
    CHECK(std::abs(camera.fx - 50.0f) < 1e-5f);
    CHECK(std::abs(camera.fy - 30.0f) < 1e-5f);
    CHECK(std::abs(camera.cx - (30.0f / 13.0f)) < 1e-5f);
    CHECK(std::abs(camera.cy - (4.0f / 3.0f)) < 1e-5f);
    checkPixel(camera.image, 0, 0, color);

    camera.releaseImageMemory();
    camera.loadImage(2.5f);
    CHECK(camera.width == 5);
    CHECK(camera.height == 3);
    CHECK(std::abs(camera.fx - 50.0f) < 1e-5f);
    CHECK(std::abs(camera.fy - 30.0f) < 1e-5f);
    CHECK(std::abs(camera.cx - (30.0f / 13.0f)) < 1e-5f);
    CHECK(std::abs(camera.cy - (4.0f / 3.0f)) < 1e-5f);

    camera.loadImage(1.5f);
    CHECK(camera.width == 8);
    CHECK(camera.height == 6);
    CHECK(std::abs(camera.fx - 80.0f) < 1e-5f);
    CHECK(std::abs(camera.fy - 60.0f) < 1e-5f);
    CHECK(std::abs(camera.cx - (48.0f / 13.0f)) < 1e-5f);
    CHECK(std::abs(camera.cy - (8.0f / 3.0f)) < 1e-5f);

    Camera orientedCalibration;
    orientedCalibration.filePath = path.string();
    orientedCalibration.width = 9;
    orientedCalibration.height = 13;
    checkThrows<msplat::InvalidDatasetError>(
        [&] { orientedCalibration.loadImage(1.0f); },
        "preserves encoded pixel coordinates");

    Camera normalizedCamera;
    normalizedCamera.filePath = path.string();
    normalizedCamera.rasterOrientation = RasterOrientation::ExifNormalized;
    normalizedCamera.width = 9;
    normalizedCamera.height = 13;
    normalizedCamera.fx = 90.0f;
    normalizedCamera.fy = 130.0f;
    // EXIF orientation 6 maps raw edge coordinates (u, v) to (H - v, u).
    // The raw principal point (6, 4) therefore becomes (5, 6), not (4, 6).
    normalizedCamera.cx = 5.0f;
    normalizedCamera.cy = 6.0f;
    normalizedCamera.loadImage(1.0f);
    CHECK(normalizedCamera.width == 9);
    CHECK(normalizedCamera.height == 13);
    CHECK(normalizedCamera.image.width == 9);
    CHECK(normalizedCamera.image.height == 13);

    normalizedCamera.loadImage(2.5f);
    CHECK(normalizedCamera.width == 3);
    CHECK(normalizedCamera.height == 5);
    CHECK(std::abs(normalizedCamera.fx - 30.0f) < 1e-5f);
    CHECK(std::abs(normalizedCamera.fy - 50.0f) < 1e-5f);
    CHECK(std::abs(normalizedCamera.cx - (5.0f / 3.0f)) < 1e-5f);
    CHECK(std::abs(normalizedCamera.cy - (30.0f / 13.0f)) < 1e-5f);

    normalizedCamera.releaseImageMemory();
    normalizedCamera.loadImage(2.5f);
    CHECK(normalizedCamera.width == 3);
    CHECK(normalizedCamera.height == 5);
    CHECK(std::abs(normalizedCamera.fx - 30.0f) < 1e-5f);
    CHECK(std::abs(normalizedCamera.fy - 50.0f) < 1e-5f);

    for (float invalid : {
             0.0f,
             0.5f,
             std::numeric_limits<float>::infinity(),
             std::numeric_limits<float>::quiet_NaN(),
         }) {
        Camera invalidCamera;
        invalidCamera.filePath = path.string();
        checkThrows<std::invalid_argument>(
            [&] { invalidCamera.loadImage(invalid); },
            "downscale factor");
    }

    Camera tooSmall;
    tooSmall.filePath = path.string();
    checkThrows<std::invalid_argument>(
        [&] { tooSmall.loadImage(32.0f); },
        "invalid width");
}

void checkBadInputs(const TempDirectory &temporary) {
    const fs::path missing = temporary.path / "missing.tiff";
    checkThrows<std::runtime_error>(
        [&] { (void)inspectImageSource(missing.string()); },
        "Failed to open image source");
    checkThrows<std::invalid_argument>(
        [&] { (void)inspectImageSource(""); },
        "Image path is empty");

    const fs::path corrupt = temporary.path / "corrupt.tiff";
    {
        std::ofstream stream(corrupt, std::ios::binary | std::ios::trunc);
        stream << "II*\0not-a-tiff";
        CHECK(static_cast<bool>(stream));
    }
    checkThrows<std::runtime_error>(
        [&] { (void)inspectImageSource(corrupt.string()); },
        {});
}

void checkImageCacheAccountingCategories() {
    Camera camera;
    camera.image.data.resize(2 * 3 * 3);
    camera.coverageMask.data.resize(2 * 3);
    Image pyramid;
    pyramid.data.resize(1 * 2 * 3);
    camera.imagePyramids.emplace(2, std::move(pyramid));
    CoverageMask maskPyramid;
    maskPyramid.data.resize(1 * 2);
    camera.coverageMaskPyramids.emplace(2, std::move(maskPyramid));
    camera.mtensorImageCache.emplace(
        2, MTensor({1, 2, 3}, DType::Float32));
    camera.mtensorCoverageMaskCache.emplace(
        2, MTensor({1, 2}, DType::UInt8));

    const size_t expectedCpuBytes =
        (2 * 3 * 3 + 1 * 2 * 3) * sizeof(float) + 2 * 3 + 1 * 2;
    const size_t expectedGpuBytes =
        1 * 2 * 3 * sizeof(float) + 1 * 2;
    CHECK(camera.cachedCpuImageBytes() == expectedCpuBytes);
    CHECK(camera.cachedGpuImageBytes() == expectedGpuBytes);
    CHECK(camera.cachedImageBytes() == expectedCpuBytes + expectedGpuBytes);

    CameraImageCache emptyCache(1.0f, 1024);
    CHECK(emptyCache.cachedCpuBytes() == 0);
    CHECK(emptyCache.cachedGpuBytes() == 0);
    CHECK(emptyCache.hitCount() == 0);
    CHECK(emptyCache.missCount() == 0);
}

void checkMaskedCacheHitAndEviction(const TempDirectory &temporary) {
    Camera resident;
    resident.filePath = "unused-on-resident-hit";
    resident.trainingMask = TrainingMaskDescriptor{
        "unused-mask-on-resident-hit", TrainingMaskChannel::Luminance};
    resident.image.width = 3;
    resident.image.height = 2;
    resident.image.data.resize(3 * 2 * 3, 0.25f);
    resident.coverageMask.width = 3;
    resident.coverageMask.height = 2;
    resident.coverageMask.data = {1, 2, 3, 4, 5, 6};
    resident.loadedImageDownscaleFactor = 1.0f;
    resident.mtensorImageCache.emplace(
        1, MTensor({2, 3, 3}, DType::Float32));
    resident.mtensorCoverageMaskCache.emplace(
        1, MTensor({2, 3}, DType::UInt8));

    std::vector<Camera> residentCameras;
    residentCameras.push_back(std::move(resident));
    CameraImageCache residentCache(1.0f, 1024);
    const CameraTrainingTarget target =
        residentCache.gpuTrainingTarget(residentCameras, 0, 1);
    CHECK(target.image != nullptr);
    CHECK(target.coverageMask != nullptr);
    CHECK(target.coverageMask->dtype() == DType::UInt8);
    CHECK(target.coverageMask->shape() == std::vector<int64_t>({2, 3}));
    CHECK(target.coverageUnits == 21);
    CHECK(residentCameras[0].coverageUnitsByDownscale.at(1) == 21);
    CHECK(residentCache.hitCount() == 1);
    CHECK(residentCache.missCount() == 0);
    CHECK(residentCache.cachedBytes() ==
          residentCameras[0].cachedImageBytes());

    const fs::path imagePath = temporary.path / "cache-rgb.tiff";
    const fs::path maskPath = temporary.path / "cache-mask.tiff";
    writeTIFF(imagePath, 4, 4, solidRGBA(4, 4, {64, 96, 128}), 1);
    writeTIFF(maskPath, 4, 4, solidRGBA(4, 4, {255, 255, 255}), 1);

    std::vector<Camera> cameras(2);
    for (Camera &camera : cameras) {
        camera.filePath = imagePath.string();
        camera.trainingMask = TrainingMaskDescriptor{
            maskPath.string(), TrainingMaskChannel::Luminance};
        camera.width = 4;
        camera.height = 4;
    }

    constexpr size_t oneCameraBytes =
        4 * 4 * 3 * sizeof(float) + 4 * 4;
    CameraImageCache evictionCache(1.0f, oneCameraBytes);
    (void)evictionCache.ensureLoaded(cameras, 0);
    CHECK(!cameras[0].image.empty());
    CHECK(!cameras[0].coverageMask.empty());
    (void)evictionCache.ensureLoaded(cameras, 1);
    CHECK(cameras[0].image.empty());
    CHECK(cameras[0].coverageMask.empty());
    CHECK(!cameras[1].image.empty());
    CHECK(!cameras[1].coverageMask.empty());
    CHECK(evictionCache.cachedCpuBytes() == oneCameraBytes);
    CHECK(evictionCache.cachedGpuBytes() == 0);

    Camera zeroAtCoarse;
    zeroAtCoarse.filePath = "unused-zero-at-coarse";
    zeroAtCoarse.trainingMask = TrainingMaskDescriptor{
        "unused-zero-mask", TrainingMaskChannel::Luminance};
    zeroAtCoarse.image.width = 4;
    zeroAtCoarse.image.height = 4;
    zeroAtCoarse.image.data.resize(4 * 4 * 3, 0.5f);
    zeroAtCoarse.coverageMask.width = 4;
    zeroAtCoarse.coverageMask.height = 4;
    zeroAtCoarse.coverageMask.data.resize(4 * 4, 0);
    zeroAtCoarse.coverageMask.data[0] = 1;
    zeroAtCoarse.loadedImageDownscaleFactor = 1.0f;
    std::vector<Camera> zeroCameras;
    zeroCameras.push_back(std::move(zeroAtCoarse));
    CameraImageCache zeroCache(1.0f, 1'024);
    checkThrows<msplat::InvalidDatasetError>(
        [&] { (void)zeroCache.gpuTrainingTarget(zeroCameras, 0, 2); },
        "zero coverage");
    CHECK(zeroCameras[0].image.empty());
    CHECK(zeroCameras[0].coverageMask.empty());
    CHECK(zeroCameras[0].imagePyramids.empty());
    CHECK(zeroCameras[0].coverageMaskPyramids.empty());
    CHECK(zeroCameras[0].mtensorImageCache.empty());
    CHECK(zeroCameras[0].mtensorCoverageMaskCache.empty());
    CHECK(zeroCameras[0].cachedImageBytes() == 0);
    CHECK(zeroCache.cachedBytes() == 0);
}

} // namespace

int main() {
    try {
        TempDirectory temporary;
        checkStage("independent PPM row order", [&] {
            checkIndependentPPMRowOrder(temporary);
        });
        checkStage("PNG write/read round trip", [&] {
            checkPNGWriteReadRoundTrip(temporary);
        });
        checkStage("EXIF metadata and raw orientation", [&] {
            checkMetadataAndRawOrientation(temporary);
        });
        checkStage("exact target size and color", [&] {
            checkExactTargetSizeAndColor(temporary);
        });
        checkStage("coverage mask decode and resize", [&] {
            checkCoverageMaskDecodeAndResize(temporary);
        });
        checkStage("coverage mask orientation", [&] {
            checkCoverageMaskOrientation(temporary);
        });
        checkStage("mirrored camera orientation rejection", [&] {
            checkMirroredCameraOrientationRejected(temporary);
        });
        checkStage("identity undistortion", [&] {
            checkIdentityUndistortion();
        });
        checkStage("undistortion raster edges", [&] {
            checkUndistortionRasterEdges();
        });
        checkStage("undistortion pixel center", [&] {
            checkUndistortionPixelCenter();
        });
        checkStage("extreme distortion rejection", [&] {
            checkExtremeDistortionRejected();
        });
        checkStage("joint mask undistortion", [&] {
            checkJointUndistortion();
        });
        checkStage("masked Camera loading", [&] {
            checkMaskedCameraLoading(temporary);
        });
        checkStage("fractional Camera scale", [&] {
            checkFractionalCameraScale(temporary);
        });
        checkStage("bad inputs", [&] {
            checkBadInputs(temporary);
        });
        checkStage("image cache accounting categories", [&] {
            checkImageCacheAccountingCategories();
        });
        checkStage("masked cache hit and eviction", [&] {
            checkMaskedCacheHitAndEviction(temporary);
        });
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "msplat image I/O test failed: " << error.what() << '\n';
        return 1;
    }
}
