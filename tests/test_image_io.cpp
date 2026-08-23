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

        if (orientation == 6) {
            const Image oriented = imreadRGB(path.string(), info, 2, 3, true);
            CHECK(oriented.width == 2);
            CHECK(oriented.height == 3);
            checkPixel(oriented, 0, 0, gridColors[3]);
            checkPixel(oriented, 1, 0, gridColors[0]);
            checkPixel(oriented, 0, 1, gridColors[4]);
            checkPixel(oriented, 1, 1, gridColors[1]);
            checkPixel(oriented, 0, 2, gridColors[5]);
            checkPixel(oriented, 1, 2, gridColors[2]);
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
    normalizedCamera.cx = 4.0f;
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
    CHECK(std::abs(normalizedCamera.cx - (4.0f / 3.0f)) < 1e-5f);
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
    Image pyramid;
    pyramid.data.resize(1 * 2 * 3);
    camera.imagePyramids.emplace(2, std::move(pyramid));
    camera.mtensorImageCache.emplace(
        2, MTensor({1, 2, 3}, DType::Float32));

    const size_t expectedCpuBytes = (2 * 3 * 3 + 1 * 2 * 3) * sizeof(float);
    const size_t expectedGpuBytes = 1 * 2 * 3 * sizeof(float);
    CHECK(camera.cachedCpuImageBytes() == expectedCpuBytes);
    CHECK(camera.cachedGpuImageBytes() == expectedGpuBytes);
    CHECK(camera.cachedImageBytes() == expectedCpuBytes + expectedGpuBytes);

    CameraImageCache emptyCache(1.0f, 1024);
    CHECK(emptyCache.cachedCpuBytes() == 0);
    CHECK(emptyCache.cachedGpuBytes() == 0);
    CHECK(emptyCache.hitCount() == 0);
    CHECK(emptyCache.missCount() == 0);
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
        checkStage("fractional Camera scale", [&] {
            checkFractionalCameraScale(temporary);
        });
        checkStage("bad inputs", [&] {
            checkBadInputs(temporary);
        });
        checkStage("image cache accounting categories", [&] {
            checkImageCacheAccountingCategories();
        });
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "msplat image I/O test failed: " << error.what() << '\n';
        return 1;
    }
}
