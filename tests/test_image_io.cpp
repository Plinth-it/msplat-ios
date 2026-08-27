#include "loaders.hpp"
#include "dataset_errors.hpp"

#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
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

void writeGrayscalePNG(const fs::path &path, int width, int height,
                       const std::vector<uint8_t> &gray) {
    CHECK(width > 0 && height > 0);
    CHECK(gray.size() == static_cast<size_t>(width) * height);

    CFHandle<CGDataProviderRef> provider(CGDataProviderCreateWithData(
        nullptr, gray.data(), gray.size(), nullptr));
    CHECK(provider);
    CFHandle<CGColorSpaceRef> colorSpace(CGColorSpaceCreateDeviceGray());
    CHECK(colorSpace);
    CFHandle<CGImageRef> image(CGImageCreate(
        static_cast<size_t>(width), static_cast<size_t>(height), 8, 8,
        static_cast<size_t>(width), colorSpace.get(), kCGImageAlphaNone,
        provider.get(), nullptr, false, kCGRenderingIntentDefault));
    CHECK(image);

    const std::string pathString = path.string();
    CFHandle<CFURLRef> url(CFURLCreateFromFileSystemRepresentation(
        nullptr, reinterpret_cast<const UInt8 *>(pathString.data()),
        static_cast<CFIndex>(pathString.size()), false));
    CHECK(url);
    CFHandle<CGImageDestinationRef> destination(
        CGImageDestinationCreateWithURL(
            url.get(), CFSTR("public.png"), 1, nullptr));
    CHECK(destination);
    CGImageDestinationAddImage(destination.get(), image.get(), nullptr);
    CHECK(CGImageDestinationFinalize(destination.get()));
}

void checkBinaryGrayscalePNGFastPathCandidate(
    const fs::path &path, int width, int height) {
    const std::string pathString = path.string();
    CFHandle<CFURLRef> url(CFURLCreateFromFileSystemRepresentation(
        nullptr, reinterpret_cast<const UInt8 *>(pathString.data()),
        static_cast<CFIndex>(pathString.size()), false));
    CHECK(url);
    CFHandle<CGImageSourceRef> source(
        CGImageSourceCreateWithURL(url.get(), nullptr));
    CHECK(source);
    CFStringRef sourceType = CGImageSourceGetType(source.get());
    CHECK(sourceType != nullptr);
    CHECK(CFEqual(sourceType, CFSTR("public.png")));

    const int64_t maximumPixelSize = std::max(width, height);
    CFHandle<CFNumberRef> maximumPixelSizeNumber(CFNumberCreate(
        nullptr, kCFNumberSInt64Type, &maximumPixelSize));
    CHECK(maximumPixelSizeNumber);
    const void *keys[] = {
        kCGImageSourceCreateThumbnailFromImageAlways,
        kCGImageSourceCreateThumbnailWithTransform,
        kCGImageSourceThumbnailMaxPixelSize,
        kCGImageSourceShouldCacheImmediately,
    };
    const void *values[] = {
        kCFBooleanTrue,
        kCFBooleanFalse,
        maximumPixelSizeNumber.get(),
        kCFBooleanTrue,
    };
    CFHandle<CFDictionaryRef> options(CFDictionaryCreate(
        nullptr, keys, values, 4,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks));
    CHECK(options);
    CFHandle<CGImageRef> thumbnail(CGImageSourceCreateThumbnailAtIndex(
        source.get(), 0, options.get()));
    CHECK(thumbnail);
    CHECK(CGImageGetWidth(thumbnail.get()) == static_cast<size_t>(width));
    CHECK(CGImageGetHeight(thumbnail.get()) == static_cast<size_t>(height));

    CGColorSpaceRef colorSpace = CGImageGetColorSpace(thumbnail.get());
    CHECK(colorSpace != nullptr);
    CHECK(CGColorSpaceGetModel(colorSpace) == kCGColorSpaceModelMonochrome);
    CFStringRef colorSpaceName = CGColorSpaceGetName(colorSpace);
    CHECK(colorSpaceName != nullptr);
    CHECK(CFEqual(colorSpaceName, kCGColorSpaceGenericGrayGamma2_2));
    CHECK(CGImageGetBitsPerComponent(thumbnail.get()) == 8);
    CHECK(CGImageGetBitsPerPixel(thumbnail.get()) == 8);
    CHECK(CGImageGetAlphaInfo(thumbnail.get()) == kCGImageAlphaNone);
    CHECK(CGImageGetDecode(thumbnail.get()) == nullptr);
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

RGB8 pixel8(const RGBA8Image &image, int x, int y) {
    CHECK(x >= 0 && x < image.width);
    CHECK(y >= 0 && y < image.height);
    const size_t index = (static_cast<size_t>(y) * image.width + x) * 4;
    return {
        image.data[index + 0],
        image.data[index + 1],
        image.data[index + 2],
    };
}

uint8_t alpha8(const RGBA8Image &image, int x, int y) {
    CHECK(x >= 0 && x < image.width);
    CHECK(y >= 0 && y < image.height);
    return image.data[
        (static_cast<size_t>(y) * image.width + x) * 4 + 3];
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

void checkPixel(const RGBA8Image &image, int x, int y, RGB8 expected,
                int tolerance = 2) {
    const RGB8 actual = pixel8(image, x, y);
    if (!near(actual, expected, tolerance)) {
        fail("RGBA8 pixel color", __LINE__,
             "at " + std::to_string(x) + "," + std::to_string(y) +
             " expected " + std::to_string(expected.red) + "," +
             std::to_string(expected.green) + "," +
             std::to_string(expected.blue) + " got " +
             std::to_string(actual.red) + "," +
             std::to_string(actual.green) + "," +
             std::to_string(actual.blue));
    }
}

RGBA8Image compactImage(const Image &source) {
    RGBA8Image compact;
    compact.width = source.width;
    compact.height = source.height;
    compact.data.resize(
        static_cast<size_t>(source.width) * source.height * 4, 255);
    for (size_t pixel = 0;
         pixel < static_cast<size_t>(source.width) * source.height;
         ++pixel) {
        for (int channel = 0; channel < 3; ++channel) {
            compact.data[pixel * 4 + channel] = static_cast<uint8_t>(
                std::clamp(
                    std::lround(source.data[pixel * 3 + channel] * 255.0f),
                    0L, 255L));
        }
    }
    return compact;
}

void checkCompactParity(const Image &expected, const RGBA8Image &actual,
                        int tolerance = 1) {
    CHECK(actual.width == expected.width);
    CHECK(actual.height == expected.height);
    CHECK(actual.data.size() ==
          static_cast<size_t>(actual.width) * actual.height * 4);
    for (size_t pixel = 0;
         pixel < static_cast<size_t>(actual.width) * actual.height;
         ++pixel) {
        for (int channel = 0; channel < 3; ++channel) {
            const int expectedByte = static_cast<int>(std::lround(
                expected.data[pixel * 3 + channel] * 255.0f));
            const int actualByte = actual.data[pixel * 4 + channel];
            CHECK(std::abs(expectedByte - actualByte) <= tolerance);
        }
        CHECK(actual.data[pixel * 4 + 3] == 255);
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

class ScopedEnvironmentVariable {
public:
    explicit ScopedEnvironmentVariable(std::string name)
        : name_(std::move(name)) {
        if (const char *value = std::getenv(name_.c_str()))
            originalValue_ = value;
    }

    ~ScopedEnvironmentVariable() {
        if (originalValue_) {
            (void)setenv(name_.c_str(), originalValue_->c_str(), 1);
        } else {
            (void)unsetenv(name_.c_str());
        }
    }

    void set(const char *value) {
        const int result = value
            ? setenv(name_.c_str(), value, 1)
            : unsetenv(name_.c_str());
        if (result != 0)
            throw std::runtime_error("could not update environment variable");
    }

private:
    std::string name_;
    std::optional<std::string> originalValue_;
};

template <typename Predicate>
void waitUntil(Predicate predicate, const std::string &description) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline)
            fail("asynchronous condition", __LINE__, description);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

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

void checkCompactRGBA8Decode(const TempDirectory &temporary) {
    const fs::path path = temporary.path / "compact-rgba8.ppm";
    std::array<uint8_t, gridColors.size() * 3> rgb{};
    for (size_t index = 0; index < gridColors.size(); ++index) {
        rgb[index * 3 + 0] = gridColors[index].red;
        rgb[index * 3 + 1] = gridColors[index].green;
        rgb[index * 3 + 2] = gridColors[index].blue;
    }
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream << "P6\n3 2\n255\n";
        stream.write(reinterpret_cast<const char *>(rgb.data()),
                     static_cast<std::streamsize>(rgb.size()));
        CHECK(static_cast<bool>(stream));
    }

    const ImageSourceInfo info = inspectImageSource(path.string());
    const RGBA8Image compact = imreadRGBA8(
        path.string(), info, 3, 2, false);
    const Image floats = imreadRGB(path.string(), info, 3, 2, false);
    CHECK(compact.width == 3);
    CHECK(compact.height == 2);
    CHECK(compact.data.size() == gridColors.size() * 4);
    for (size_t index = 0; index < gridColors.size(); ++index) {
        const int x = static_cast<int>(index % 3);
        const int y = static_cast<int>(index / 3);
        checkPixel(compact, x, y, gridColors[index], 0);
        CHECK(alpha8(compact, x, y) == 255);
        const RGB8 floatPixel = pixel8(floats, x, y);
        CHECK(floatPixel.red == compact.data[index * 4 + 0]);
        CHECK(floatPixel.green == compact.data[index * 4 + 1]);
        CHECK(floatPixel.blue == compact.data[index * 4 + 2]);
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

CoverageMask referenceResizeCoverageArea(
    const CoverageMask &src, int dstW, int dstH) {
    CoverageMask dst;
    dst.width = dstW;
    dst.height = dstH;
    dst.data.resize(static_cast<size_t>(dstW) * dstH);

    const double scaleX = static_cast<double>(src.width) / dstW;
    const double scaleY = static_cast<double>(src.height) / dstH;
    for (int dy = 0; dy < dstH; ++dy) {
        const double srcY0 = dy * scaleY;
        const double srcY1 = (dy + 1) * scaleY;
        for (int dx = 0; dx < dstW; ++dx) {
            const double srcX0 = dx * scaleX;
            const double srcX1 = (dx + 1) * scaleX;
            double sum = 0.0;
            double totalArea = 0.0;

            const int iy0 = static_cast<int>(srcY0);
            const int iy1 = std::min(
                static_cast<int>(std::ceil(srcY1)), src.height);
            const int ix0 = static_cast<int>(srcX0);
            const int ix1 = std::min(
                static_cast<int>(std::ceil(srcX1)), src.width);
            for (int iy = iy0; iy < iy1; ++iy) {
                const double wy =
                    std::min(static_cast<double>(iy + 1), srcY1) -
                    std::max(static_cast<double>(iy), srcY0);
                for (int ix = ix0; ix < ix1; ++ix) {
                    const double wx =
                        std::min(static_cast<double>(ix + 1), srcX1) -
                        std::max(static_cast<double>(ix), srcX0);
                    const double area = wx * wy;
                    const size_t sourceIndex =
                        static_cast<size_t>(iy) * src.width + ix;
                    sum += src.data[sourceIndex] * area;
                    totalArea += area;
                }
            }

            const long rounded = std::lround(sum / totalArea);
            dst.data[static_cast<size_t>(dy) * dstW + dx] =
                static_cast<uint8_t>(std::clamp(rounded, 0L, 255L));
        }
    }
    return dst;
}

void checkCoverageAreaResizeParity() {
    std::mt19937 random(0x4d534b31u);
    std::uniform_int_distribution<int> byteValue(0, 255);
    const auto checkCase = [&](int srcW, int srcH, int dstW, int dstH) {
        CoverageMask source;
        source.width = srcW;
        source.height = srcH;
        source.data.resize(static_cast<size_t>(srcW) * srcH);
        for (uint8_t &value : source.data)
            value = static_cast<uint8_t>(byteValue(random));

        const CoverageMask expected =
            referenceResizeCoverageArea(source, dstW, dstH);
        const CoverageMask actual = resizeCoverageArea(source, dstW, dstH);
        CHECK(actual.width == expected.width);
        CHECK(actual.height == expected.height);
        CHECK(actual.data == expected.data);
    };

    constexpr std::array<std::array<int, 4>, 8> fixedCases = {{
        {{1, 1, 1, 1}},
        {{2, 2, 1, 1}},
        {{7, 5, 4, 3}},
        {{13, 11, 17, 19}},
        {{33, 17, 16, 8}},
        {{64, 48, 23, 19}},
        {{302, 403, 120, 160}},
        {{403, 302, 160, 120}},
    }};
    for (const auto &dimensions : fixedCases) {
        checkCase(
            dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    }

    std::uniform_int_distribution<int> sourceExtent(1, 37);
    std::uniform_int_distribution<int> destinationExtent(1, 41);
    for (int iteration = 0; iteration < 300; ++iteration) {
        checkCase(
            sourceExtent(random), sourceExtent(random),
            destinationExtent(random), destinationExtent(random));
    }
}

void checkBinaryGrayscalePNGMaskDecode(const TempDirectory &temporary) {
    constexpr int width = 7;
    constexpr int height = 5;
    const std::vector<uint8_t> binary = {
        0,   0,   255, 255, 255, 0,   0,
        0,   255, 255, 255, 0,   0,   255,
        255, 255, 0,   0,   0,   255, 255,
        255, 0,   0,   255, 255, 255, 0,
        0,   0,   255, 0,   255, 0,   255,
    };
    const fs::path binaryPath = temporary.path / "binary-gray-mask.png";
    writeGrayscalePNG(binaryPath, width, height, binary);
    checkBinaryGrayscalePNGFastPathCandidate(binaryPath, width, height);
    const ImageSourceInfo binaryInfo = inspectImageSource(binaryPath.string());

    const CoverageMask full = imreadCoverageMask(
        binaryPath.string(), binaryInfo, width, height, false,
        TrainingMaskChannel::Automatic);
    CHECK(full.width == width);
    CHECK(full.height == height);
    CHECK(full.data == binary);

    // Requesting orientation normalization deliberately bypasses the fast
    // path. Orientation 1 should remain byte-identical through the fallback.
    const CoverageMask transformed = imreadCoverageMask(
        binaryPath.string(), binaryInfo, width, height, true,
        TrainingMaskChannel::Automatic);
    CHECK(transformed.data == binary);

    CoverageMask sourceMask;
    sourceMask.width = width;
    sourceMask.height = height;
    sourceMask.data = binary;
    const CoverageMask expected = resizeCoverageArea(sourceMask, 4, 3);
    const CoverageMask scaled = imreadCoverageMask(
        binaryPath.string(), binaryInfo, 4, 3, false,
        TrainingMaskChannel::Automatic);
    CHECK(scaled.width == 4);
    CHECK(scaled.height == 3);
    CHECK(scaled.data == expected.data);
    CHECK(buildCoverageRenderTileMap(scaled).data ==
          buildCoverageRenderTileMap(expected).data);

    // Explicit luminance deliberately takes the established RGBA fallback.
    const CoverageMask luminance = imreadCoverageMask(
        binaryPath.string(), binaryInfo, 4, 3, false,
        TrainingMaskChannel::Luminance);
    CHECK(luminance.data == scaled.data);

    const std::vector<uint8_t> soft = {0, 1, 64, 127, 128, 254, 255};
    const fs::path softPath = temporary.path / "soft-gray-mask.png";
    writeGrayscalePNG(softPath, 7, 1, soft);
    const ImageSourceInfo softInfo = inspectImageSource(softPath.string());
    const CoverageMask automaticSoft = imreadCoverageMask(
        softPath.string(), softInfo, 7, 1, false,
        TrainingMaskChannel::Automatic);
    const CoverageMask luminanceSoft = imreadCoverageMask(
        softPath.string(), softInfo, 7, 1, false,
        TrainingMaskChannel::Luminance);
    CHECK(automaticSoft.data == luminanceSoft.data);
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

void checkCompactResizeAndUndistortionParity() {
    const Image source = coordinateImage(9, 9);
    const RGBA8Image compact = compactImage(source);

    const Image resized = resizeArea(source, 4, 3);
    const RGBA8Image compactResized = resizeRGBA8Area(compact, 4, 3);
    checkCompactParity(resized, compactResized);

    const auto undistorted = undistortImage(
        source, 10.0f, 10.0f, 4.5f, 4.5f,
        0.04f, -0.005f, 0.002f, -0.003f, 0.0f);
    const auto compactUndistorted = undistortRGBA8Image(
        compact, 10.0f, 10.0f, 4.5f, 4.5f,
        0.04f, -0.005f, 0.002f, -0.003f, 0.0f);
    CHECK(compactUndistorted.width == undistorted.width);
    CHECK(compactUndistorted.height == undistorted.height);
    CHECK(compactUndistorted.fx == undistorted.fx);
    CHECK(compactUndistorted.fy == undistorted.fy);
    CHECK(compactUndistorted.cx == undistorted.cx);
    CHECK(compactUndistorted.cy == undistorted.cy);
    checkCompactParity(undistorted.image, compactUndistorted.image);
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
    const auto compactResult = undistortRGBA8ImageAndCoverageMask(
        compactImage(image), mask, 10.0f, 10.0f, 6.0f, 4.0f,
        0.08f, -0.01f, 0.002f, -0.003f, 0.0f);
    CHECK(result.image.width == result.width);
    CHECK(result.image.height == result.height);
    CHECK(result.coverageMask.width == result.width);
    CHECK(result.coverageMask.height == result.height);
    CHECK(compactResult.width == result.width);
    CHECK(compactResult.height == result.height);
    CHECK(compactResult.fx == result.fx);
    CHECK(compactResult.fy == result.fy);
    CHECK(compactResult.cx == result.cx);
    CHECK(compactResult.cy == result.cy);
    CHECK(compactResult.coverageMask.data == result.coverageMask.data);
    checkCompactParity(result.image, compactResult.image);
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

std::vector<uint8_t> repeatingGridRGBA(int width, int height) {
    std::vector<RGB8> colors(static_cast<size_t>(width) * height);
    for (size_t index = 0; index < colors.size(); ++index)
        colors[index] = gridColors[index % gridColors.size()];
    return rgbaGrid(width, height, colors);
}

Camera prefetchTestCamera(
    const fs::path &imagePath, int width, int height,
    std::optional<TrainingMaskDescriptor> trainingMask = std::nullopt) {
    Camera camera;
    camera.filePath = imagePath.string();
    camera.width = width;
    camera.height = height;
    camera.fx = static_cast<float>(width);
    camera.fy = static_cast<float>(height);
    camera.cx = static_cast<float>(width) * 0.5f;
    camera.cy = static_cast<float>(height) * 0.5f;
    camera.trainingMask = std::move(trainingMask);
    return camera;
}

struct TrainingTargetSnapshot {
    std::vector<int64_t> imageShape;
    std::vector<uint8_t> imageBytes;
    std::optional<std::vector<int64_t>> maskShape;
    std::optional<std::vector<uint8_t>> maskBytes;
    bool maskPackedInImage = false;
    std::optional<std::vector<int64_t>> coverageRenderTileShape;
    std::optional<std::vector<uint8_t>> coverageRenderTileBytes;
    uint64_t coverageUnits = 0;

    size_t byteCount() const {
        return imageBytes.size() +
            (maskBytes && !maskPackedInImage ? maskBytes->size() : 0) +
            (coverageRenderTileBytes
                ? coverageRenderTileBytes->size() : 0);
    }
};

TrainingTargetSnapshot snapshotTarget(const CameraTrainingTarget &target) {
    CHECK(target.image != nullptr);
    CHECK(target.image->dtype() == DType::UInt8);

    TrainingTargetSnapshot result;
    result.imageShape = target.image->shape();
    const uint8_t *imageBytes = target.image->data<uint8_t>();
    result.imageBytes.assign(
        imageBytes, imageBytes + target.image->nbytes());
    if (target.coverageMask) {
        CHECK(target.coverageMask->dtype() == DType::UInt8);
        result.maskPackedInImage = target.coverageMask == target.image;
        if (result.maskPackedInImage) {
            CHECK(target.image->shape().size() == 3);
            result.maskShape = std::vector<int64_t>{
                target.image->size(0), target.image->size(1)};
            result.maskBytes.emplace();
            result.maskBytes->reserve(
                static_cast<size_t>(target.image->size(0) *
                                    target.image->size(1)));
            for (size_t pixel = 0;
                 pixel < target.image->nbytes() / 4u; ++pixel) {
                result.maskBytes->push_back(imageBytes[pixel * 4u + 3u]);
            }
        } else {
            result.maskShape = target.coverageMask->shape();
            const uint8_t *maskBytes = target.coverageMask->data<uint8_t>();
            result.maskBytes.emplace(
                maskBytes, maskBytes + target.coverageMask->nbytes());
        }
    }
    if (target.coverageRenderTiles) {
        CHECK(target.coverageRenderTiles->dtype() == DType::UInt8);
        result.coverageRenderTileShape =
            target.coverageRenderTiles->shape();
        const uint8_t *tileBytes =
            target.coverageRenderTiles->data<uint8_t>();
        result.coverageRenderTileBytes.emplace(
            tileBytes, tileBytes + target.coverageRenderTiles->nbytes());
    }
    result.coverageUnits = target.coverageUnits;
    return result;
}

void checkSameTarget(const TrainingTargetSnapshot &expected,
                     const TrainingTargetSnapshot &actual) {
    CHECK(actual.imageShape == expected.imageShape);
    CHECK(actual.imageBytes == expected.imageBytes);
    CHECK(actual.maskShape == expected.maskShape);
    CHECK(actual.maskBytes == expected.maskBytes);
    CHECK(actual.maskPackedInImage == expected.maskPackedInImage);
    CHECK(actual.coverageRenderTileShape ==
          expected.coverageRenderTileShape);
    CHECK(actual.coverageRenderTileBytes ==
          expected.coverageRenderTileBytes);
    CHECK(actual.coverageUnits == expected.coverageUnits);
}

void checkCoverageRenderTileMap() {
    CoverageMask mask;
    mask.width = 33;
    mask.height = 17;
    mask.data.assign(static_cast<size_t>(mask.width) * mask.height, 0);

    // A soft nonzero center at x=20 reaches x=15 through the exact five-pixel
    // SSIM halo, activating both adjacent 16-pixel tiles.
    mask.data[8u * mask.width + 20u] = 1;
    CoverageMask tiles = buildCoverageRenderTileMap(mask);
    CHECK(tiles.width == 3);
    CHECK(tiles.height == 2);
    CHECK(tiles.data == std::vector<uint8_t>({1, 1, 0, 0, 0, 0}));

    // At x=21 the left tile is six pixels away and must remain inactive.
    std::fill(mask.data.begin(), mask.data.end(), 0);
    mask.data[8u * mask.width + 21u] = 255;
    tiles = buildCoverageRenderTileMap(mask);
    CHECK(tiles.data == std::vector<uint8_t>({0, 1, 0, 0, 0, 0}));

    // A bottom-right edge center clamps safely and reaches four tiles.
    std::fill(mask.data.begin(), mask.data.end(), 0);
    mask.data[16u * mask.width + 32u] = 255;
    tiles = buildCoverageRenderTileMap(mask);
    CHECK(tiles.data == std::vector<uint8_t>({0, 1, 1, 0, 1, 1}));

    std::fill(mask.data.begin(), mask.data.end(), 255);
    tiles = buildCoverageRenderTileMap(mask);
    CHECK(std::all_of(tiles.data.begin(), tiles.data.end(),
                      [](uint8_t value) { return value == 1; }));
}

void checkPrefetchEnvironmentOptIn() {
    const char *originalValue = std::getenv("MSPLAT_CAMERA_PREFETCH");
    const std::optional<std::string> original = originalValue
        ? std::optional<std::string>(originalValue)
        : std::nullopt;

    {
        ScopedEnvironmentVariable environment("MSPLAT_CAMERA_PREFETCH");
        environment.set(nullptr);
        CHECK(!CameraImageCache::defaultPrefetchEnabled());

        for (const char *disabled : {
                 "", "0", "01", "true", "TRUE", " 1", "1 ", "2",
             }) {
            environment.set(disabled);
            CHECK(!CameraImageCache::defaultPrefetchEnabled());
        }

        environment.set("1");
        CHECK(CameraImageCache::defaultPrefetchEnabled());
        CameraImageCache defaulted(1.0f, 1'024);
        CHECK(defaulted.prefetchEnabled());
        CameraImageCache explicitlyDisabled(1.0f, 1'024, false);
        CHECK(!explicitlyDisabled.prefetchEnabled());
        explicitlyDisabled.enablePrefetch();
        CHECK(explicitlyDisabled.prefetchEnabled());
        explicitlyDisabled.enablePrefetch();
        CHECK(explicitlyDisabled.prefetchEnabled());
    }

    const char *restoredValue = std::getenv("MSPLAT_CAMERA_PREFETCH");
    if (original) {
        CHECK(restoredValue != nullptr);
        CHECK(*original == restoredValue);
    } else {
        CHECK(restoredValue == nullptr);
    }
}

void checkUnmaskedPrefetchParity(const TempDirectory &temporary) {
    constexpr int width = 6;
    constexpr int height = 4;
    constexpr int stageDownscale = 2;
    const fs::path imagePath = temporary.path / "prefetch-unmasked.tiff";
    writeTIFF(
        imagePath, width, height, repeatingGridRGBA(width, height), 1);

    std::vector<Camera> synchronousCameras;
    synchronousCameras.push_back(
        prefetchTestCamera(imagePath, width, height));
    CameraImageCache synchronousCache(1.0f, 1'024 * 1'024, false);
    const TrainingTargetSnapshot expected = snapshotTarget(
        synchronousCache.gpuTrainingTarget(
            synchronousCameras, 0, stageDownscale));
    CHECK(expected.maskBytes == std::nullopt);

    std::vector<Camera> cameras;
    cameras.push_back(prefetchTestCamera(imagePath, width, height));
    CameraImageCache cache(1.0f, 1'024 * 1'024, true);
    cache.prefetchTrainingTarget(cameras, 0, stageDownscale);
    CHECK(cache.prefetchScheduledCount() == 1);
    CHECK(cache.prefetchUsedCount() == 0);
    CHECK(cache.prefetchDiscardedCount() == 0);

    waitUntil(
        [&] { return cache.cachedCpuBytes() == expected.byteCount(); },
        "unmasked camera prefetch did not finish");

    // The worker owns only its source snapshot and staging Camera. The live
    // camera is untouched until foreground upload and publication.
    const Camera &live = cameras[0];
    CHECK(!live.declared.captured);
    CHECK(live.width == width);
    CHECK(live.height == height);
    CHECK(live.fx == static_cast<float>(width));
    CHECK(live.fy == static_cast<float>(height));
    CHECK(live.loadedImageDownscaleFactor == 0.0f);
    CHECK(live.image.empty());
    CHECK(live.coverageMask.empty());
    CHECK(live.imagePyramids.empty());
    CHECK(live.coverageMaskPyramids.empty());
    CHECK(live.mtensorImageCache.empty());
    CHECK(live.coverageUnitsByDownscale.empty());
    CHECK(live.cachedImageBytes() == 0);

    const TrainingTargetSnapshot actual = snapshotTarget(
        cache.gpuTrainingTarget(cameras, 0, stageDownscale));
    checkSameTarget(expected, actual);
    CHECK(cache.prefetchScheduledCount() == 1);
    CHECK(cache.prefetchUsedCount() == 1);
    CHECK(cache.prefetchDiscardedCount() == 0);
    CHECK(cache.missCount() == 1);
    CHECK(cache.hitCount() == 0);
    CHECK(cache.cachedCpuBytes() == 0);
    CHECK(cache.cachedGpuBytes() == expected.byteCount());
}

void checkMaskedPrefetchParity(const TempDirectory &temporary) {
    constexpr int width = 6;
    constexpr int height = 4;
    constexpr int stageDownscale = 2;
    const fs::path imagePath = temporary.path / "prefetch-masked-rgb.tiff";
    const fs::path maskPath = temporary.path / "prefetch-masked-mask.tiff";
    writeTIFF(
        imagePath, width, height, repeatingGridRGBA(width, height), 1);

    std::vector<uint8_t> maskRGBA(
        static_cast<size_t>(width) * height * 4, 0);
    for (size_t pixel = 0; pixel < maskRGBA.size() / 4; ++pixel)
        maskRGBA[pixel * 4 + 3] = static_cast<uint8_t>(32 + pixel * 7);
    writeTIFF(maskPath, width, height, maskRGBA, 1);
    const TrainingMaskDescriptor mask{
        maskPath.string(), TrainingMaskChannel::Alpha};

    std::vector<Camera> synchronousCameras;
    synchronousCameras.push_back(
        prefetchTestCamera(imagePath, width, height, mask));
    CameraImageCache synchronousCache(1.0f, 1'024 * 1'024, false);
    const TrainingTargetSnapshot expected = snapshotTarget(
        synchronousCache.gpuTrainingTarget(
            synchronousCameras, 0, stageDownscale));
    CHECK(expected.maskBytes.has_value());
    CHECK(expected.maskPackedInImage);
    CHECK(expected.coverageRenderTileBytes.has_value());
    CHECK(expected.byteCount() ==
          static_cast<size_t>(width / stageDownscale) *
              (height / stageDownscale) * 4u + 1u);

    std::vector<Camera> cameras;
    cameras.push_back(prefetchTestCamera(imagePath, width, height, mask));
    CameraImageCache cache(1.0f, 1'024 * 1'024, true);
    cache.prefetchTrainingTarget(cameras, 0, stageDownscale);
    waitUntil(
        [&] { return cache.cachedCpuBytes() == expected.byteCount(); },
        "masked camera prefetch did not finish");

    const TrainingTargetSnapshot actual = snapshotTarget(
        cache.gpuTrainingTarget(cameras, 0, stageDownscale));
    checkSameTarget(expected, actual);
    CHECK(cache.prefetchScheduledCount() == 1);
    CHECK(cache.prefetchUsedCount() == 1);
    CHECK(cache.prefetchDiscardedCount() == 0);
    CHECK(cache.missCount() == 1);
    CHECK(cache.cachedCpuBytes() == 0);
    CHECK(cache.cachedGpuBytes() == expected.byteCount());

    // A prepared masked target keeps the denominator's source identity after
    // decoded pixels are released. Removing both files proves the next cache
    // hit cannot silently re-decode them.
    CHECK(fs::remove(imagePath));
    CHECK(fs::remove(maskPath));
    const TrainingTargetSnapshot residentHit = snapshotTarget(
        cache.gpuTrainingTarget(cameras, 0, stageDownscale));
    checkSameTarget(expected, residentHit);
    CHECK(cache.hitCount() == 1);
    CHECK(cache.missCount() == 1);
    CHECK(cache.cachedCpuBytes() == 0);
}

void checkTransparentMaskedTargetCapability(const TempDirectory &temporary) {
    constexpr int width = 33;
    constexpr int height = 17;
    const fs::path imagePath =
        temporary.path / "prefetch-transparent-rgb.tiff";
    const fs::path maskPath =
        temporary.path / "prefetch-transparent-mask.tiff";
    const std::vector<uint8_t> imageRGBA =
        repeatingGridRGBA(width, height);
    std::vector<uint8_t> maskRGBA(
        static_cast<size_t>(width) * height * 4u, 0);
    maskRGBA[(8u * width + 20u) * 4u + 3u] = 1;
    maskRGBA[(static_cast<size_t>(height - 1) * width + width - 1) *
             4u + 3u] = 255;
    writeTIFF(imagePath, width, height, imageRGBA, 1);
    writeTIFF(maskPath, width, height, maskRGBA, 1);
    const TrainingMaskDescriptor mask{
        maskPath.string(), TrainingMaskChannel::Alpha};

    std::vector<Camera> coverageCameras;
    coverageCameras.push_back(
        prefetchTestCamera(imagePath, width, height, mask));
    CameraImageCache coverageCache(1.0f, 1'024 * 1'024, false);
    const TrainingTargetSnapshot expectedCoverage = snapshotTarget(
        coverageCache.gpuTrainingTarget(coverageCameras, 0, 1, true));
    CHECK(expectedCoverage.coverageRenderTileBytes ==
          std::optional<std::vector<uint8_t>>(
              std::vector<uint8_t>({1, 1, 1, 0, 1, 1})));

    std::vector<Camera> transparentCameras;
    transparentCameras.push_back(
        prefetchTestCamera(imagePath, width, height, mask));
    CameraImageCache transparentCache(1.0f, 1'024 * 1'024, false);
    const TrainingTargetSnapshot expectedTransparent = snapshotTarget(
        transparentCache.gpuTrainingTarget(
            transparentCameras, 0, 1, false));
    CHECK(expectedTransparent.imageBytes == expectedCoverage.imageBytes);
    CHECK(expectedTransparent.maskBytes == expectedCoverage.maskBytes);
    CHECK(expectedTransparent.coverageUnits == expectedCoverage.coverageUnits);
    CHECK(!expectedTransparent.coverageRenderTileBytes.has_value());

    std::vector<Camera> cameras;
    cameras.push_back(prefetchTestCamera(imagePath, width, height, mask));
    CameraImageCache cache(1.0f, 1'024 * 1'024, true);
    cache.prefetchTrainingTarget(cameras, 0, 1, false);
    waitUntil(
        [&] { return cache.cachedCpuBytes() == expectedTransparent.byteCount(); },
        "transparent masked camera prefetch did not finish");
    const TrainingTargetSnapshot prefetchedTransparent = snapshotTarget(
        cache.gpuTrainingTarget(cameras, 0, 1, false));
    checkSameTarget(expectedTransparent, prefetchedTransparent);
    CHECK(cache.prefetchScheduledCount() == 1);
    CHECK(cache.prefetchUsedCount() == 1);
    CHECK(cache.prefetchWaitCount() <= cache.prefetchUsedCount());
    CHECK(cache.cachedGpuBytes() == expectedTransparent.byteCount());

    CHECK(fs::remove(imagePath));
    CHECK(fs::remove(maskPath));
    const TrainingTargetSnapshot sourceFreeHit = snapshotTarget(
        cache.gpuTrainingTarget(cameras, 0, 1, false));
    checkSameTarget(expectedTransparent, sourceFreeHit);
    CHECK(cache.hitCount() == 1);
    CHECK(cache.missCount() == 1);

    // A tile-less Transparent target cannot satisfy a later Coverage request.
    // Recreate the sources, upgrade the resident capability, then verify that
    // the Coverage superset can still serve Transparent without exposing its
    // retained tile map.
    writeTIFF(imagePath, width, height, imageRGBA, 1);
    writeTIFF(maskPath, width, height, maskRGBA, 1);
    const TrainingTargetSnapshot upgradedCoverage = snapshotTarget(
        cache.gpuTrainingTarget(cameras, 0, 1, true));
    checkSameTarget(expectedCoverage, upgradedCoverage);
    CHECK(cache.hitCount() == 1);
    CHECK(cache.missCount() == 2);
    CHECK(cache.cachedGpuBytes() == expectedCoverage.byteCount());

    const TrainingTargetSnapshot coverageSupersetHit = snapshotTarget(
        cache.gpuTrainingTarget(cameras, 0, 1, false));
    checkSameTarget(expectedTransparent, coverageSupersetHit);
    CHECK(cache.hitCount() == 2);
    CHECK(cache.missCount() == 2);
}

void checkNonmatchingTargetPreservesPrefetch(
    const TempDirectory &temporary) {
    constexpr int width = 6;
    constexpr int height = 4;
    const fs::path prefetchedPath =
        temporary.path / "prefetch-preserved.tiff";
    const fs::path foregroundPath =
        temporary.path / "prefetch-foreground.tiff";
    writeTIFF(
        prefetchedPath, width, height,
        repeatingGridRGBA(width, height), 1);
    writeTIFF(
        foregroundPath, width, height,
        solidRGBA(width, height, {19, 73, 211}), 1);

    std::vector<Camera> cameras;
    cameras.push_back(prefetchTestCamera(prefetchedPath, width, height));
    cameras.push_back(prefetchTestCamera(foregroundPath, width, height));
    CameraImageCache cache(1.0f, 1'024 * 1'024, true);

    const size_t stagedBytes = static_cast<size_t>(width) * height * 4;
    cache.prefetchTrainingTarget(cameras, 0, 1);
    waitUntil(
        [&] { return cache.cachedCpuBytes() == stagedBytes; },
        "preserved camera prefetch did not finish");

    const TrainingTargetSnapshot foreground = snapshotTarget(
        cache.gpuTrainingTarget(cameras, 1, 1));
    CHECK(!foreground.imageBytes.empty());
    CHECK(cache.prefetchScheduledCount() == 1);
    CHECK(cache.prefetchUsedCount() == 0);
    CHECK(cache.prefetchDiscardedCount() == 0);
    CHECK(cache.cachedCpuBytes() == stagedBytes);

    const TrainingTargetSnapshot prefetched = snapshotTarget(
        cache.gpuTrainingTarget(cameras, 0, 1));
    CHECK(!prefetched.imageBytes.empty());
    CHECK(cache.prefetchScheduledCount() == 1);
    CHECK(cache.prefetchUsedCount() == 1);
    CHECK(cache.prefetchDiscardedCount() == 0);
    CHECK(cache.missCount() == 2);
    CHECK(cache.cachedCpuBytes() == 0);
}

void checkFailedPrefetchDiscard(const TempDirectory &temporary) {
    constexpr int width = 6;
    constexpr int height = 4;
    const fs::path validPath = temporary.path / "prefetch-after-failure.tiff";
    writeTIFF(validPath, width, height,
              repeatingGridRGBA(width, height), 1);

    std::vector<Camera> cameras;
    cameras.push_back(prefetchTestCamera(
        temporary.path / "missing-prefetch.tiff", width, height));
    cameras.push_back(prefetchTestCamera(validPath, width, height));
    CameraImageCache cache(1.0f, 1'024 * 1'024, true);

    cache.prefetchTrainingTarget(cameras, 0, 1);
    CHECK(cache.prefetchScheduledCount() == 1);
    // This joins the worker and suppresses its missing-file exception.
    cache.discardPrefetch();
    CHECK(cache.prefetchDiscardedCount() == 1);
    CHECK(cache.prefetchUsedCount() == 0);
    CHECK(cache.cachedCpuBytes() == 0);
    cache.discardPrefetch();
    CHECK(cache.prefetchDiscardedCount() == 1);

    // A discarded failure leaves the depth-one slot reusable.
    const size_t stagedBytes = static_cast<size_t>(width) * height * 4;
    cache.prefetchTrainingTarget(cameras, 1, 1);
    waitUntil(
        [&] { return cache.cachedCpuBytes() == stagedBytes; },
        "camera prefetch slot was not reusable after discard");
    const TrainingTargetSnapshot target = snapshotTarget(
        cache.gpuTrainingTarget(cameras, 1, 1));
    CHECK(!target.imageBytes.empty());
    CHECK(cache.prefetchScheduledCount() == 2);
    CHECK(cache.prefetchUsedCount() == 1);
    CHECK(cache.prefetchDiscardedCount() == 1);
    CHECK(cache.cachedCpuBytes() == 0);
}

void checkImageCacheAccountingCategories() {
    Camera camera;
    camera.image.data.resize(2 * 3 * 4);
    camera.coverageMask.data.resize(2 * 3);
    RGBA8Image pyramid;
    pyramid.data.resize(1 * 2 * 4);
    camera.imagePyramids.emplace(2, std::move(pyramid));
    CoverageMask maskPyramid;
    maskPyramid.data.resize(1 * 2);
    camera.coverageMaskPyramids.emplace(2, std::move(maskPyramid));
    camera.mtensorImageCache.emplace(
        2, MTensor({1, 2, 4}, DType::UInt8));
    camera.mtensorCoverageRenderTileCache.emplace(
        2, MTensor({1, 2}, DType::UInt8));

    const size_t expectedCpuBytes =
        2 * 3 * 4 + 1 * 2 * 4 + 2 * 3 + 1 * 2;
    const size_t expectedGpuBytes = 1 * 2 * 4 + 1 * 2;
    CHECK(camera.cachedCpuImageBytes() == expectedCpuBytes);
    CHECK(camera.cachedGpuImageBytes() == expectedGpuBytes);
    CHECK(camera.cachedImageBytes() == expectedCpuBytes + expectedGpuBytes);

    CameraImageCache emptyCache(1.0f, 1024);
    CHECK(emptyCache.cachedCpuBytes() == 0);
    CHECK(emptyCache.cachedGpuBytes() == 0);
    CHECK(emptyCache.hitCount() == 0);
    CHECK(emptyCache.missCount() == 0);
}

void checkCompactGPUTrainingTargetUpload(const TempDirectory &temporary) {
    const fs::path path = temporary.path / "compact-target-upload.ppm";
    std::array<uint8_t, gridColors.size() * 3> rgb{};
    for (size_t index = 0; index < gridColors.size(); ++index) {
        rgb[index * 3 + 0] = gridColors[index].red;
        rgb[index * 3 + 1] = gridColors[index].green;
        rgb[index * 3 + 2] = gridColors[index].blue;
    }
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream << "P6\n3 2\n255\n";
        stream.write(reinterpret_cast<const char *>(rgb.data()),
                     static_cast<std::streamsize>(rgb.size()));
        CHECK(static_cast<bool>(stream));
    }

    const ImageSourceInfo info = inspectImageSource(path.string());
    const RGBA8Image expected = imreadRGBA8(
        path.string(), info, 3, 2, false);

    Camera camera;
    camera.filePath = path.string();
    camera.width = 3;
    camera.height = 2;
    std::vector<Camera> cameras;
    cameras.push_back(std::move(camera));
    CameraImageCache cache(1.0f, 1'024);

    const CameraTrainingTarget first =
        cache.gpuTrainingTarget(cameras, 0, 1);
    CHECK(first.image != nullptr);
    CHECK(first.coverageMask == nullptr);
    CHECK(first.coverageRenderTiles == nullptr);
    CHECK(first.coverageUnits == 3u * 2u * 255u);
    CHECK(first.image->isGpu());
    CHECK(first.image->dtype() == DType::UInt8);
    CHECK(first.image->shape() == std::vector<int64_t>({2, 3, 4}));
    CHECK(first.image->nbytes() == expected.data.size());
    CHECK(std::equal(
        expected.data.begin(), expected.data.end(),
        first.image->data<uint8_t>()));
    CHECK(cameras[0].image.empty());
    CHECK(cameras[0].imagePyramids.empty());
    CHECK(cache.cachedCpuBytes() == 0);
    CHECK(cache.cachedGpuBytes() == expected.data.size());
    CHECK(cache.hitCount() == 0);
    CHECK(cache.missCount() == 1);

    // A resident compact target must not touch the source file again.
    CHECK(fs::remove(path));
    const CameraTrainingTarget second =
        cache.gpuTrainingTarget(cameras, 0, 1);
    CHECK(second.image == first.image);
    CHECK(std::equal(
        expected.data.begin(), expected.data.end(),
        second.image->data<uint8_t>()));
    CHECK(cache.cachedCpuBytes() == 0);
    CHECK(cache.cachedGpuBytes() == expected.data.size());
    CHECK(cache.hitCount() == 1);
    CHECK(cache.missCount() == 1);
}

void checkCompactTrainingTargetStorageValidation() {
    auto makeCamera = [](size_t imageBytes) {
        Camera camera;
        camera.filePath = "manual-rgba8";
        camera.width = 2;
        camera.height = 2;
        camera.image.width = 2;
        camera.image.height = 2;
        camera.image.data.resize(imageBytes, 64);
        return camera;
    };

    for (const size_t imageBytes : {size_t{15}, size_t{17}}) {
        Camera camera = makeCamera(imageBytes);
        checkThrows<msplat::InvalidDatasetError>(
            [&] { (void)camera.getGPUTrainingTarget(1); },
            "Training image storage does not match its dimensions");
        CHECK(camera.mtensorImageCache.empty());
        CHECK(camera.mtensorCoverageRenderTileCache.empty());
    }

    for (const size_t maskBytes : {size_t{3}, size_t{5}}) {
        Camera camera = makeCamera(16);
        camera.trainingMask = TrainingMaskDescriptor{
            "manual-mask", TrainingMaskChannel::Luminance};
        camera.decodedTrainingMaskSource = camera.trainingMask;
        camera.coverageMask.width = 2;
        camera.coverageMask.height = 2;
        camera.coverageMask.data.resize(maskBytes, 255);
        // Exercise the upload boundary directly: a manually populated
        // denominator bypasses getCoverageUnits' normal storage validation.
        camera.coverageUnitsByDownscale.emplace(1, 4u * 255u);

        checkThrows<msplat::InvalidDatasetError>(
            [&] { (void)camera.getGPUTrainingTarget(1); },
            "Training mask storage does not match its dimensions");
        CHECK(camera.mtensorImageCache.empty());
        CHECK(camera.mtensorCoverageRenderTileCache.empty());
    }
}

void checkMaskedCacheHitAndEviction(const TempDirectory &temporary) {
    Camera resident;
    resident.filePath = "unused-on-resident-hit";
    resident.trainingMask = TrainingMaskDescriptor{
        "unused-mask-on-resident-hit", TrainingMaskChannel::Luminance};
    resident.decodedTrainingMaskSource = resident.trainingMask;
    resident.image.width = 3;
    resident.image.height = 2;
    resident.image.data.resize(3 * 2 * 4, 64);
    for (size_t pixel = 0; pixel < 3 * 2; ++pixel)
        resident.image.data[pixel * 4 + 3] = 255;
    resident.coverageMask.width = 3;
    resident.coverageMask.height = 2;
    resident.coverageMask.data = {1, 2, 3, 4, 5, 6};
    resident.loadedImageDownscaleFactor = 1.0f;
    resident.mtensorImageCache.emplace(
        1, MTensor({2, 3, 4}, DType::UInt8));
    resident.mtensorCoverageRenderTileCache.emplace(
        1, MTensor({1, 1}, DType::UInt8));
    resident.mtensorCoverageRenderTileCache.at(1).data<uint8_t>()[0] = 1;
    resident.gpuTrainingMaskSourceByDownscale.emplace(
        1, resident.trainingMask);
    resident.coverageUnitsByDownscale.emplace(1, 21);
    for (size_t pixel = 0; pixel < resident.coverageMask.data.size(); ++pixel)
        resident.mtensorImageCache.at(1).data<uint8_t>()[pixel * 4u + 3u] =
            resident.coverageMask.data[pixel];

    std::vector<Camera> residentCameras;
    residentCameras.push_back(std::move(resident));
    CameraImageCache residentCache(1.0f, 1024);
    const CameraTrainingTarget target =
        residentCache.gpuTrainingTarget(residentCameras, 0, 1);
    CHECK(target.image != nullptr);
    CHECK(target.image->dtype() == DType::UInt8);
    CHECK(target.image->shape() == std::vector<int64_t>({2, 3, 4}));
    CHECK(target.coverageMask == target.image);
    CHECK(target.coverageRenderTiles ==
          &residentCameras[0].mtensorCoverageRenderTileCache.at(1));
    CHECK(target.image->nbytes() == 2u * 3u * 4u);
    CHECK(target.coverageUnits == 21);
    CHECK(residentCameras[0].coverageUnitsByDownscale.at(1) == 21);
    CHECK(residentCache.hitCount() == 1);
    CHECK(residentCache.missCount() == 0);
    CHECK(residentCache.cachedBytes() ==
          residentCameras[0].cachedImageBytes());
    CHECK(residentCameras[0].image.empty());
    CHECK(residentCameras[0].coverageMask.empty());
    CHECK(residentCache.cachedCpuBytes() == 0);

    const CameraTrainingTarget secondTarget =
        residentCache.gpuTrainingTarget(residentCameras, 0, 1);
    CHECK(secondTarget.image == target.image);
    CHECK(secondTarget.coverageRenderTiles == target.coverageRenderTiles);
    CHECK(residentCache.hitCount() == 2);
    CHECK(residentCache.missCount() == 0);
    CHECK(residentCache.cachedCpuBytes() == 0);

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

    constexpr size_t oneCameraBytes = 4 * 4 * 4 + 4 * 4;
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
    zeroAtCoarse.decodedTrainingMaskSource = zeroAtCoarse.trainingMask;
    zeroAtCoarse.image.width = 4;
    zeroAtCoarse.image.height = 4;
    zeroAtCoarse.image.data.resize(4 * 4 * 4, 128);
    for (size_t pixel = 0; pixel < 4 * 4; ++pixel)
        zeroAtCoarse.image.data[pixel * 4 + 3] = 255;
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
    CHECK(zeroCameras[0].mtensorCoverageRenderTileCache.empty());
    CHECK(zeroCameras[0].gpuTrainingMaskSourceByDownscale.empty());
    CHECK(zeroCameras[0].cachedImageBytes() == 0);
    CHECK(zeroCache.cachedBytes() == 0);
}

void checkTrainingMaskSourceMutation(const TempDirectory &temporary) {
    constexpr int width = 33;
    constexpr int height = 17;
    const fs::path imagePath = temporary.path / "mask-mutation-image.tiff";
    const fs::path maskAPath = temporary.path / "mask-mutation-a.tiff";
    const fs::path maskBPath = temporary.path / "mask-mutation-b.tiff";
    writeTIFF(imagePath, width, height,
              solidRGBA(width, height, {80, 120, 160}), 1);

    std::vector<uint8_t> maskA(static_cast<size_t>(width) * height * 4, 0);
    maskA[3] = 64;
    writeTIFF(maskAPath, width, height, maskA, 1);
    std::vector<uint8_t> maskB(static_cast<size_t>(width) * height * 4, 0);
    const size_t lastPixel = static_cast<size_t>(width) * height - 1;
    maskB[lastPixel * 4 + 3] = 255;
    writeTIFF(maskBPath, width, height, maskB, 1);

    // Decode A only on CPU, verify a repeated load is a true cache hit, then
    // mutate the descriptor before any GPU target has been published. The next
    // target request must reload B rather than reuse A's bytes or denominator.
    Camera decoded = prefetchTestCamera(
        imagePath, width, height,
        TrainingMaskDescriptor{
            maskAPath.string(), TrainingMaskChannel::Alpha});
    decoded.loadImage(1.0f);
    CHECK(decoded.getCoverageUnits(1) == 64);
    const uint8_t *decodedImageBytes = decoded.image.data.data();
    const uint8_t *decodedMaskBytes = decoded.coverageMask.data.data();
    decoded.loadImage(1.0f);
    CHECK(decoded.image.data.data() == decodedImageBytes);
    CHECK(decoded.coverageMask.data.data() == decodedMaskBytes);
    decoded.trainingMask = TrainingMaskDescriptor{
        maskBPath.string(), TrainingMaskChannel::Alpha};
    std::vector<Camera> decodedCameras;
    decodedCameras.push_back(std::move(decoded));
    CameraImageCache decodedCache(1.0f, 1'024 * 1'024, false);
    Camera &reloadedForNewMask = decodedCache.ensureLoaded(
        decodedCameras, 0);
    CHECK(reloadedForNewMask.getCoverageUnits(1) == 255);
    const TrainingTargetSnapshot decodedMutation = snapshotTarget(
        decodedCache.gpuTrainingTarget(decodedCameras, 0, 1));
    CHECK(decodedMutation.coverageUnits == 255);
    CHECK(decodedMutation.coverageRenderTileBytes ==
          std::optional<std::vector<uint8_t>>(
              std::vector<uint8_t>({0, 1, 1, 0, 1, 1})));

    std::vector<Camera> cameras;
    cameras.push_back(prefetchTestCamera(imagePath, width, height));
    CameraImageCache cache(1.0f, 1'024 * 1'024, false);

    const CameraTrainingTarget unmasked =
        cache.gpuTrainingTarget(cameras, 0, 1);
    CHECK(unmasked.coverageMask == nullptr);
    CHECK(unmasked.coverageRenderTiles == nullptr);

    cameras[0].trainingMask = TrainingMaskDescriptor{
        maskAPath.string(), TrainingMaskChannel::Alpha};
    const TrainingTargetSnapshot firstMask = snapshotTarget(
        cache.gpuTrainingTarget(cameras, 0, 1));
    CHECK(firstMask.coverageUnits == 64);
    CHECK(firstMask.coverageRenderTileBytes ==
          std::optional<std::vector<uint8_t>>(
              std::vector<uint8_t>({1, 0, 0, 0, 0, 0})));

    cameras[0].trainingMask = TrainingMaskDescriptor{
        maskBPath.string(), TrainingMaskChannel::Alpha};
    const TrainingTargetSnapshot secondMask = snapshotTarget(
        cache.gpuTrainingTarget(cameras, 0, 1));
    CHECK(secondMask.coverageUnits == 255);
    CHECK(secondMask.coverageRenderTileBytes ==
          std::optional<std::vector<uint8_t>>(
              std::vector<uint8_t>({0, 1, 1, 0, 1, 1})));
    CHECK(cache.hitCount() == 0);
    CHECK(cache.missCount() == 3);
}

void checkCameraPoseCacheContract() {
    Camera camera;
    const std::array<float, 16> identity = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };

    camera.setCameraToWorld(identity.data());
    CHECK(!camera.projectionCacheMatchesPose());
    camera.recordProjectionCachePose();
    CHECK(camera.projectionCacheMatchesPose());

    camera.cachedViewMat = MTensor({4, 4}, DType::Float32);
    camera.cachedProjViewMat = MTensor({4, 4}, DType::Float32);
    CHECK(camera.cachedViewMat.defined());
    CHECK(camera.cachedProjViewMat.defined());
    std::array<float, 16> translated = identity;
    translated[3] = 0.125f;
    camera.setCameraToWorld(translated.data());
    CHECK(!camera.cachedViewMat.defined());
    CHECK(!camera.cachedProjViewMat.defined());
    CHECK(!camera.projectionCacheMatchesPose());
    camera.recordProjectionCachePose();

    // Even a legacy direct write must force prepareCam to rebuild its derived
    // matrices instead of reusing a view from the previous pose.
    camera.camToWorld[3] = 0.25f;
    CHECK(!camera.projectionCacheMatchesPose());
    camera.recordProjectionCachePose();
    CHECK(camera.projectionCacheMatchesPose());

    camera.invalidateProjectionCache();
    CHECK(!camera.projectionCacheMatchesPose());

    std::array<float, 16> invalid = identity;
    invalid[6] = std::numeric_limits<float>::quiet_NaN();
    checkThrows<std::invalid_argument>(
        [&] { camera.setCameraToWorld(invalid.data()); }, "finite");
}

} // namespace

int main() {
    try {
        TempDirectory temporary;
        checkStage("independent PPM row order", [&] {
            checkIndependentPPMRowOrder(temporary);
        });
        checkStage("compact RGBA8 decode", [&] {
            checkCompactRGBA8Decode(temporary);
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
        checkStage("coverage area resize parity", [&] {
            checkCoverageAreaResizeParity();
        });
        checkStage("binary grayscale PNG mask decode", [&] {
            checkBinaryGrayscalePNGMaskDecode(temporary);
        });
        checkStage("coverage render-tile halo", [&] {
            checkCoverageRenderTileMap();
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
        checkStage("compact resize and undistortion parity", [&] {
            checkCompactResizeAndUndistortionParity();
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
        checkStage("camera prefetch environment opt-in", [&] {
            checkPrefetchEnvironmentOptIn();
        });
        checkStage("image cache accounting categories", [&] {
            checkImageCacheAccountingCategories();
        });
        checkStage("compact GPU training-target upload", [&] {
            checkCompactGPUTrainingTargetUpload(temporary);
        });
        checkStage("compact training-target storage validation", [&] {
            checkCompactTrainingTargetStorageValidation();
        });
        checkStage("masked cache hit and eviction", [&] {
            checkMaskedCacheHitAndEviction(temporary);
        });
        checkStage("training-mask source mutation", [&] {
            checkTrainingMaskSourceMutation(temporary);
        });
        checkStage("unmasked camera prefetch parity", [&] {
            checkUnmaskedPrefetchParity(temporary);
        });
        checkStage("masked camera prefetch parity", [&] {
            checkMaskedPrefetchParity(temporary);
        });
        checkStage("transparent masked target capability", [&] {
            checkTransparentMaskedTargetCapability(temporary);
        });
        checkStage("nonmatching target preserves camera prefetch", [&] {
            checkNonmatchingTargetPreservesPrefetch(temporary);
        });
        checkStage("failed camera prefetch discard", [&] {
            checkFailedPrefetchDiscard(temporary);
        });
        checkStage("camera pose cache contract", [&] {
            checkCameraPoseCacheContract();
        });
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "msplat image I/O test failed: " << error.what() << '\n';
        return 1;
    }
}
