#include "loaders.hpp"
#include "dataset_errors.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

// ── Image loading (CoreGraphics) ─────────────────────────────────────────────

namespace {

template <typename T>
class CFHandle {
public:
    explicit CFHandle(T value = nullptr) : value_(value) {}
    ~CFHandle() { reset(); }

    CFHandle(const CFHandle &) = delete;
    CFHandle &operator=(const CFHandle &) = delete;

    CFHandle(CFHandle &&other) noexcept : value_(other.value_) {
        other.value_ = nullptr;
    }

    CFHandle &operator=(CFHandle &&other) noexcept {
        if (this != &other) {
            reset();
            value_ = other.value_;
            other.value_ = nullptr;
        }
        return *this;
    }

    T get() const { return value_; }
    explicit operator bool() const { return value_ != nullptr; }

    void reset(T value = nullptr) {
        if (value_) CFRelease(value_);
        value_ = value;
    }

private:
    T value_;
};

size_t checkedElementCount(int width, int height, size_t channels,
                           size_t elementBytes,
                           const std::string &context) {
    if (width <= 0 || height <= 0 || channels == 0 || elementBytes == 0) {
        throw std::runtime_error(context + " has invalid dimensions " +
                                 std::to_string(width) + "x" +
                                 std::to_string(height));
    }

    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    if (w > std::numeric_limits<size_t>::max() / h) {
        throw std::overflow_error(context + " pixel count overflows size_t");
    }
    const size_t pixels = w * h;
    if (pixels > std::numeric_limits<size_t>::max() / channels) {
        throw std::overflow_error(context + " element count overflows size_t");
    }
    const size_t elements = pixels * channels;
    if (elements > std::numeric_limits<size_t>::max() / elementBytes) {
        throw std::overflow_error(context + " byte count overflows size_t");
    }
    return elements;
}

CFHandle<CFURLRef> makeFileURL(const std::string &path) {
    if (path.empty() ||
        path.size() > static_cast<size_t>(std::numeric_limits<CFIndex>::max())) {
        throw std::invalid_argument("Image path is empty or too long");
    }

    CFHandle<CFURLRef> url(CFURLCreateFromFileSystemRepresentation(
        nullptr,
        reinterpret_cast<const UInt8 *>(path.data()),
        static_cast<CFIndex>(path.size()),
        false));
    if (!url)
        throw msplat::DatasetIOError("Failed to create image URL: " + path);
    return url;
}

CFHandle<CGImageSourceRef> openImageSource(const std::string &path) {
    auto url = makeFileURL(path);
    CFHandle<CGImageSourceRef> source(CGImageSourceCreateWithURL(url.get(), nullptr));
    if (!source || CGImageSourceGetCount(source.get()) == 0) {
        throw msplat::DatasetIOError("Failed to open image source: " + path);
    }
    return source;
}

int readPositiveInt(CFDictionaryRef properties, CFStringRef key,
                    const char *label, const std::string &path) {
    const void *value = CFDictionaryGetValue(properties, key);
    if (!value || CFGetTypeID(value) != CFNumberGetTypeID()) {
        throw msplat::InvalidDatasetError(
            "Image " + std::string(label) + " is missing: " + path);
    }

    int64_t number = 0;
    if (!CFNumberGetValue(static_cast<CFNumberRef>(value),
                          kCFNumberSInt64Type, &number) ||
        number <= 0 || number > std::numeric_limits<int>::max()) {
        throw msplat::InvalidDatasetError(
            "Image " + std::string(label) + " is invalid: " + path);
    }
    return static_cast<int>(number);
}

ImageSourceInfo sourceInfo(CGImageSourceRef source, const std::string &path) {
    CFHandle<CFDictionaryRef> properties(
        CGImageSourceCopyPropertiesAtIndex(source, 0, nullptr));
    if (!properties) {
        throw msplat::DatasetIOError(
            "Failed to decode image metadata: " + path);
    }

    ImageSourceInfo info;
    info.rawWidth = readPositiveInt(properties.get(), kCGImagePropertyPixelWidth,
                                    "pixel width", path);
    info.rawHeight = readPositiveInt(properties.get(), kCGImagePropertyPixelHeight,
                                     "pixel height", path);

    if (const void *value = CFDictionaryGetValue(
            properties.get(), kCGImagePropertyOrientation)) {
        if (CFGetTypeID(value) != CFNumberGetTypeID()) {
            throw msplat::InvalidDatasetError(
                "Image EXIF orientation is invalid: " + path);
        }
        int32_t orientation = 0;
        if (!CFNumberGetValue(static_cast<CFNumberRef>(value),
                              kCFNumberSInt32Type, &orientation) ||
            orientation < 1 || orientation > 8) {
            throw msplat::InvalidDatasetError(
                "Image EXIF orientation must be in 1...8: " + path);
        }
        info.exifOrientation = orientation;
    }

    const bool swapsDimensions = info.exifOrientation >= 5;
    info.orientedWidth = swapsDimensions ? info.rawHeight : info.rawWidth;
    info.orientedHeight = swapsDimensions ? info.rawWidth : info.rawHeight;
    return info;
}

bool sameSourceInfo(const ImageSourceInfo &lhs, const ImageSourceInfo &rhs) {
    return lhs.rawWidth == rhs.rawWidth &&
           lhs.rawHeight == rhs.rawHeight &&
           lhs.orientedWidth == rhs.orientedWidth &&
           lhs.orientedHeight == rhs.orientedHeight &&
           lhs.exifOrientation == rhs.exifOrientation;
}

bool hasAlphaChannel(CGImageAlphaInfo alphaInfo) {
    switch (alphaInfo) {
        case kCGImageAlphaPremultipliedLast:
        case kCGImageAlphaPremultipliedFirst:
        case kCGImageAlphaLast:
        case kCGImageAlphaFirst:
        case kCGImageAlphaOnly:
            return true;
        case kCGImageAlphaNone:
        case kCGImageAlphaNoneSkipLast:
        case kCGImageAlphaNoneSkipFirst:
            return false;
    }
    return false;
}

CFHandle<CGImageRef> createThumbnail(
    CGImageSourceRef source, const std::string &path,
    const ImageSourceInfo &expectedSourceInfo,
    int targetWidth, int targetHeight, bool applyExifOrientation) {
    const ImageSourceInfo currentSourceInfo = sourceInfo(source, path);
    if (!sameSourceInfo(currentSourceInfo, expectedSourceInfo)) {
        throw msplat::DatasetChangedError(
            "Image dimensions or orientation changed while loading: " + path);
    }

    const int64_t maximumPixelSize = std::max(targetWidth, targetHeight);
    CFHandle<CFNumberRef> maximumPixelSizeNumber(CFNumberCreate(
        nullptr, kCFNumberSInt64Type, &maximumPixelSize));
    if (!maximumPixelSizeNumber) {
        throw msplat::DatasetIOError(
            "Failed to configure image thumbnail: " + path);
    }

    const void *keys[] = {
        kCGImageSourceCreateThumbnailFromImageAlways,
        kCGImageSourceCreateThumbnailWithTransform,
        kCGImageSourceThumbnailMaxPixelSize,
        kCGImageSourceShouldCacheImmediately,
    };
    const void *values[] = {
        kCFBooleanTrue,
        applyExifOrientation ? kCFBooleanTrue : kCFBooleanFalse,
        maximumPixelSizeNumber.get(),
        kCFBooleanTrue,
    };
    CFHandle<CFDictionaryRef> options(CFDictionaryCreate(
        nullptr, keys, values, 4,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks));
    if (!options) {
        throw msplat::DatasetIOError(
            "Failed to configure image thumbnail: " + path);
    }

    CFHandle<CGImageRef> thumbnail(CGImageSourceCreateThumbnailAtIndex(
        source, 0, options.get()));
    if (!thumbnail || CGImageGetWidth(thumbnail.get()) == 0 ||
        CGImageGetHeight(thumbnail.get()) == 0) {
        throw msplat::DatasetIOError(
            "Failed to decode image thumbnail: " + path);
    }
    return thumbnail;
}

} // namespace

ImageSourceInfo inspectImageSource(const std::string &path) {
    auto source = openImageSource(path);
    return sourceInfo(source.get(), path);
}

Image imreadRGB(const std::string &path, const ImageSourceInfo &expectedSourceInfo,
                int targetWidth, int targetHeight,
                bool applyExifOrientation) {
    const size_t rgbaElements = checkedElementCount(
        targetWidth, targetHeight, 4, sizeof(uint8_t), "Decoded image");
    const size_t rgbElements = checkedElementCount(
        targetWidth, targetHeight, 3, sizeof(float), "Decoded image");
    auto source = openImageSource(path);
    auto thumbnail = createThumbnail(
        source.get(), path, expectedSourceInfo, targetWidth, targetHeight,
        applyExifOrientation);
    const size_t rowBytes = static_cast<size_t>(targetWidth) * 4;

    // ImageIO performs the codec-aware downsample and, when requested by a
    // calibration-aware caller, the EXIF transform. Rendering the thumbnail
    // once more guarantees the exact truncation-sized canvas expected by the
    // training plan (codec rounding can differ by one pixel).
    std::vector<uint8_t> rgba(rgbaElements);
    CFHandle<CGColorSpaceRef> colorSpace(
        CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
    if (!colorSpace) {
        throw msplat::DatasetIOError(
            "Failed to create image decode color space: " + path);
    }

    const CGBitmapInfo bitmapInfo = static_cast<CGBitmapInfo>(
        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big);
    CFHandle<CGContextRef> context(CGBitmapContextCreate(
        rgba.data(), static_cast<size_t>(targetWidth),
        static_cast<size_t>(targetHeight), 8, rowBytes, colorSpace.get(),
        bitmapInfo));
    if (!context) {
        throw msplat::DatasetIOError(
            "Failed to allocate image decode context: " + path);
    }

    CGContextSetBlendMode(context.get(), kCGBlendModeCopy);
    CGContextSetInterpolationQuality(context.get(), kCGInterpolationHigh);
    CGContextDrawImage(
        context.get(),
        CGRectMake(0, 0, static_cast<CGFloat>(targetWidth),
                   static_cast<CGFloat>(targetHeight)),
        thumbnail.get());

    Image image;
    image.width = targetWidth;
    image.height = targetHeight;
    image.data.resize(rgbElements);
    const size_t pixelCount = rgbElements / 3;
    for (size_t i = 0; i < pixelCount; ++i) {
        image.data[i * 3 + 0] = rgba[i * 4 + 0] / 255.0f;
        image.data[i * 3 + 1] = rgba[i * 4 + 1] / 255.0f;
        image.data[i * 3 + 2] = rgba[i * 4 + 2] / 255.0f;
    }
    return image;
}

CoverageMask imreadCoverageMask(
    const std::string &path, const ImageSourceInfo &expectedSourceInfo,
    int targetWidth, int targetHeight, bool applyExifOrientation,
    TrainingMaskChannel channel) {
    if (channel != TrainingMaskChannel::Luminance &&
        channel != TrainingMaskChannel::Alpha) {
        throw std::invalid_argument("Unknown training mask channel");
    }

    const int sourceWidth = applyExifOrientation
        ? expectedSourceInfo.orientedWidth : expectedSourceInfo.rawWidth;
    const int sourceHeight = applyExifOrientation
        ? expectedSourceInfo.orientedHeight : expectedSourceInfo.rawHeight;
    const size_t rgbaElements = checkedElementCount(
        sourceWidth, sourceHeight, 4, sizeof(uint8_t),
        "Decoded training mask");
    const size_t sourceMaskElements = checkedElementCount(
        sourceWidth, sourceHeight, 1, sizeof(uint8_t),
        "Decoded training mask");
    (void)checkedElementCount(
        targetWidth, targetHeight, 1, sizeof(uint8_t),
        "Decoded training mask target");

    std::vector<uint8_t> rgba(rgbaElements);
    {
        auto source = openImageSource(path);
        auto thumbnail = createThumbnail(
            source.get(), path, expectedSourceInfo, sourceWidth, sourceHeight,
            applyExifOrientation);
        if (CGImageGetWidth(thumbnail.get()) !=
                static_cast<size_t>(sourceWidth) ||
            CGImageGetHeight(thumbnail.get()) !=
                static_cast<size_t>(sourceHeight)) {
            throw msplat::DatasetIOError(
                "Training mask full-resolution decode changed dimensions: " +
                path);
        }
        if (channel == TrainingMaskChannel::Alpha &&
            !hasAlphaChannel(CGImageGetAlphaInfo(thumbnail.get()))) {
            throw msplat::InvalidDatasetError(
                "Training mask requests alpha coverage but the image has no "
                "alpha channel: " + path);
        }

        CFHandle<CGColorSpaceRef> colorSpace(
            CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
        if (!colorSpace) {
            throw msplat::DatasetIOError(
                "Failed to create training mask color space: " + path);
        }

        // A premultiplied destination makes luminance coverage naturally
        // include source alpha instead of treating transparent RGB as visible.
        const CGBitmapInfo bitmapInfo = static_cast<CGBitmapInfo>(
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        const size_t rowBytes = static_cast<size_t>(sourceWidth) * 4;
        CFHandle<CGContextRef> context(CGBitmapContextCreate(
            rgba.data(), static_cast<size_t>(sourceWidth),
            static_cast<size_t>(sourceHeight), 8, rowBytes, colorSpace.get(),
            bitmapInfo));
        if (!context) {
            throw msplat::DatasetIOError(
                "Failed to allocate training mask decode context: " + path);
        }

        CGContextSetBlendMode(context.get(), kCGBlendModeCopy);
        CGContextSetInterpolationQuality(context.get(), kCGInterpolationNone);
        CGContextDrawImage(
            context.get(),
            CGRectMake(0, 0, static_cast<CGFloat>(sourceWidth),
                       static_cast<CGFloat>(sourceHeight)),
            thumbnail.get());

        // Compact coverage into the start of the RGBA buffer. This is safe
        // in-place because each destination byte precedes every unread source
        // pixel, and it avoids a fifth source-sized allocation while the
        // decoded CGImage is resident.
        for (size_t i = 0; i < sourceMaskElements; ++i) {
            if (channel == TrainingMaskChannel::Alpha) {
                rgba[i] = rgba[i * 4 + 3];
                continue;
            }
            const float luminance =
                0.2126f * rgba[i * 4 + 0] +
                0.7152f * rgba[i * 4 + 1] +
                0.0722f * rgba[i * 4 + 2];
            rgba[i] = static_cast<uint8_t>(
                std::clamp(std::lround(luminance), 0L, 255L));
        }
    }

    CoverageMask mask;
    mask.width = sourceWidth;
    mask.height = sourceHeight;
    mask.data.assign(rgba.begin(), rgba.begin() + sourceMaskElements);
    if (sourceWidth == targetWidth && sourceHeight == targetHeight) {
        return mask;
    }

    // Coverage is derived before any resampling, then reduced exactly once by
    // the explicit box filter. ImageIO does not document its thumbnail filter,
    // so target-sized thumbnail decode would silently alter sparse coverage.
    return resizeCoverageArea(mask, targetWidth, targetHeight);
}

// ── Image writing (CoreGraphics PNG) ─────────────────────────────────────────

void imwriteRGB(const std::string &path, const Image &img) {
    const size_t rgbElements = checkedElementCount(
        img.width, img.height, 3, sizeof(float), "Image to write");
    if (img.data.size() != rgbElements) {
        throw std::invalid_argument(
            "Image storage does not match its dimensions");
    }
    const size_t rgbaElements = checkedElementCount(
        img.width, img.height, 4, sizeof(uint8_t), "Image to write");
    const size_t rowBytes = static_cast<size_t>(img.width) * 4;

    std::vector<uint8_t> rgba(rgbaElements);
    const size_t width = static_cast<size_t>(img.width);
    const size_t height = static_cast<size_t>(img.height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            const size_t sourcePixel = y * width + x;
            for (size_t channel = 0; channel < 3; ++channel) {
                const float value = img.data[sourcePixel * 3 + channel];
                const float byteValue = std::isfinite(value)
                    ? std::clamp(value * 255.0f, 0.0f, 255.0f)
                    : 0.0f;
                rgba[sourcePixel * 4 + channel] =
                    static_cast<uint8_t>(byteValue + 0.5f);
            }
            rgba[sourcePixel * 4 + 3] = 255;
        }
    }

    CFHandle<CGColorSpaceRef> colorSpace(
        CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
    if (!colorSpace) {
        throw std::runtime_error("Failed to create sRGB color space");
    }
    const CGBitmapInfo bitmapInfo = static_cast<CGBitmapInfo>(
        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big);
    CFHandle<CGContextRef> context(CGBitmapContextCreate(
        rgba.data(), static_cast<size_t>(img.width),
        static_cast<size_t>(img.height), 8, rowBytes, colorSpace.get(),
        bitmapInfo));
    if (!context) {
        throw std::runtime_error("Failed to create image write context: " + path);
    }

    CFHandle<CGImageRef> image(CGBitmapContextCreateImage(context.get()));
    if (!image) {
        throw std::runtime_error("Failed to create image for writing: " + path);
    }
    auto url = makeFileURL(path);
    CFHandle<CGImageDestinationRef> destination(
        CGImageDestinationCreateWithURL(url.get(), CFSTR("public.png"), 1,
                                        nullptr));
    if (!destination) {
        throw std::runtime_error("Failed to create PNG destination: " + path);
    }
    CGImageDestinationAddImage(destination.get(), image.get(), nullptr);
    if (!CGImageDestinationFinalize(destination.get())) {
        throw std::runtime_error("Failed to write PNG: " + path);
    }
}

// ── Area-based image resize (box filter) ─────────────────────────────────────

Image resizeArea(const Image &src, int dstW, int dstH) {
    const size_t sourceElements = checkedElementCount(
        src.width, src.height, 3, sizeof(float), "Source image");
    if (src.data.size() != sourceElements) {
        throw std::invalid_argument(
            "Source image storage does not match its dimensions");
    }
    const size_t destinationElements = checkedElementCount(
        dstW, dstH, 3, sizeof(float), "Resized image");

    Image dst;
    dst.width = dstW;
    dst.height = dstH;
    dst.data.resize(destinationElements, 0.0f);

    float scaleX = (float)src.width / dstW;
    float scaleY = (float)src.height / dstH;

    for (int dy = 0; dy < dstH; dy++) {
        float srcY0 = dy * scaleY;
        float srcY1 = (dy + 1) * scaleY;

        for (int dx = 0; dx < dstW; dx++) {
            float srcX0 = dx * scaleX;
            float srcX1 = (dx + 1) * scaleX;

            float sum[3] = {};
            float totalArea = 0;

            int iy0 = (int)srcY0;
            int iy1 = std::min((int)std::ceil(srcY1), src.height);
            int ix0 = (int)srcX0;
            int ix1 = std::min((int)std::ceil(srcX1), src.width);

            for (int iy = iy0; iy < iy1; iy++) {
                float wy = std::min((float)(iy + 1), srcY1) - std::max((float)iy, srcY0);
                for (int ix = ix0; ix < ix1; ix++) {
                    float wx = std::min((float)(ix + 1), srcX1) - std::max((float)ix, srcX0);
                    float area = wx * wy;
                    const size_t sourceIndex =
                        (static_cast<size_t>(iy) * src.width + ix) * 3;
                    const float *p = &src.data[sourceIndex];
                    sum[0] += p[0] * area;
                    sum[1] += p[1] * area;
                    sum[2] += p[2] * area;
                    totalArea += area;
                }
            }

            const size_t destinationIndex =
                (static_cast<size_t>(dy) * dstW + dx) * 3;
            float *out = &dst.data[destinationIndex];
            float inv = 1.0f / totalArea;
            out[0] = sum[0] * inv;
            out[1] = sum[1] * inv;
            out[2] = sum[2] * inv;
        }
    }
    return dst;
}

// ── Undistortion (Brown-Conrady model) ───────────────────────────────────────

CoverageMask resizeCoverageArea(const CoverageMask &src, int dstW, int dstH) {
    const size_t sourceElements = checkedElementCount(
        src.width, src.height, 1, sizeof(uint8_t), "Source training mask");
    if (src.data.size() != sourceElements) {
        throw std::invalid_argument(
            "Source training mask storage does not match its dimensions");
    }
    const size_t destinationElements = checkedElementCount(
        dstW, dstH, 1, sizeof(uint8_t), "Resized training mask");

    CoverageMask dst;
    dst.width = dstW;
    dst.height = dstH;
    dst.data.resize(destinationElements);

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

// Apply forward distortion: normalized undistorted → normalized distorted
static void distortPoint(float x, float y,
    float k1, float k2, float p1, float p2, float k3,
    float &xd, float &yd)
{
    float r2 = x * x + y * y;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
    xd = x * radial + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
    yd = y * radial + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y;
}

// Iteratively invert distortion: normalized distorted → normalized undistorted
static void undistortPoint(float xd, float yd,
    float k1, float k2, float p1, float p2, float k3,
    float &xu, float &yu)
{
    xu = xd;
    yu = yd;
    for (int i = 0; i < 20; i++) {
        float r2 = xu * xu + yu * yu;
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
        float dx = 2.0f * p1 * xu * yu + p2 * (r2 + 2.0f * xu * xu);
        float dy = p1 * (r2 + 2.0f * yu * yu) + 2.0f * p2 * xu * yu;
        xu = (xd - dx) / radial;
        yu = (yd - dy) / radial;
    }
}

// Bilinear sample from float32 image, returns pixel value at (x, y)
static void bilinearSample(const Image &img, float x, float y, float out[3]) {
    int x0 = (int)std::floor(x);
    int y0 = (int)std::floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp to image bounds
    x0 = std::clamp(x0, 0, img.width - 1);
    x1 = std::clamp(x1, 0, img.width - 1);
    y0 = std::clamp(y0, 0, img.height - 1);
    y1 = std::clamp(y1, 0, img.height - 1);

    float fx = x - std::floor(x);
    float fy = y - std::floor(y);

    const float *p00 = &img.data[(y0 * img.width + x0) * 3];
    const float *p10 = &img.data[(y0 * img.width + x1) * 3];
    const float *p01 = &img.data[(y1 * img.width + x0) * 3];
    const float *p11 = &img.data[(y1 * img.width + x1) * 3];

    for (int c = 0; c < 3; c++) {
        float top    = p00[c] * (1.0f - fx) + p10[c] * fx;
        float bottom = p01[c] * (1.0f - fx) + p11[c] * fx;
        out[c] = top * (1.0f - fy) + bottom * fy;
    }
}

static uint8_t bilinearSampleCoverage(
    const CoverageMask &mask, float x, float y) {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = std::clamp(x0, 0, mask.width - 1);
    x1 = std::clamp(x1, 0, mask.width - 1);
    y0 = std::clamp(y0, 0, mask.height - 1);
    y1 = std::clamp(y1, 0, mask.height - 1);

    const float fractionX = x - std::floor(x);
    const float fractionY = y - std::floor(y);
    const auto sample = [&](int sampleX, int sampleY) {
        return static_cast<float>(mask.data[
            static_cast<size_t>(sampleY) * mask.width + sampleX]);
    };
    const float top =
        sample(x0, y0) * (1.0f - fractionX) +
        sample(x1, y0) * fractionX;
    const float bottom =
        sample(x0, y1) * (1.0f - fractionX) +
        sample(x1, y1) * fractionX;
    return static_cast<uint8_t>(std::clamp(
        std::lround(top * (1.0f - fractionY) + bottom * fractionY),
        0L, 255L));
}

UndistortResult undistortImage(const Image &src,
    float fx, float fy, float cx, float cy,
    float k1, float k2, float p1, float p2, float k3)
{
    int w = src.width, h = src.height;

    // Find valid region by undistorting boundary points of the source image.
    // For each point on the distorted boundary, find its undistorted position.
    // The inner rectangle of all undistorted boundary points = valid region (alpha=0).
    float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
    int nSamples = 200;
    for (int i = 0; i < nSamples; i++) {
        float t = (float)i / (nSamples - 1);
        // Four edges of the distorted image
        float edges[][2] = {
            {t * w, 0.0f},          // top
            {t * w, (float)(h-1)},  // bottom
            {0.0f, t * h},          // left
            {(float)(w-1), t * h},  // right
        };
        for (auto &pt : edges) {
            float xd = (pt[0] - cx) / fx;
            float yd = (pt[1] - cy) / fy;
            float xu, yu;
            undistortPoint(xd, yd, k1, k2, p1, p2, k3, xu, yu);
            // Back to pixel coords using original intrinsics as the "new" camera
            float pu = xu * fx + cx;
            float pv = yu * fy + cy;
            minX = std::min(minX, pu);
            maxX = std::max(maxX, pu);
            minY = std::min(minY, pv);
            maxY = std::max(maxY, pv);
        }
    }

    // Inner rectangle: clamp to image bounds and take the inner edges
    // For top/left edges: take the max (inner boundary)
    // For bottom/right edges: take the min (inner boundary)
    // But we need to separate inner from outer per edge...
    // Top edge gives us maxY from top → that's minY constraint
    // Bottom edge gives us minY from bottom → that's maxY constraint
    // Actually, let me resample per-edge:
    float topMax = -1e9f, bottomMin = 1e9f, leftMax = -1e9f, rightMin = 1e9f;
    for (int i = 0; i < nSamples; i++) {
        float t = (float)i / (nSamples - 1);

        // Top edge: all points along y=0
        float xd = (t * w - cx) / fx, yd = (0.0f - cy) / fy;
        float xu, yu;
        undistortPoint(xd, yd, k1, k2, p1, p2, k3, xu, yu);
        topMax = std::max(topMax, yu * fy + cy);

        // Bottom edge: all points along y=h-1
        xd = (t * w - cx) / fx; yd = ((float)(h-1) - cy) / fy;
        undistortPoint(xd, yd, k1, k2, p1, p2, k3, xu, yu);
        bottomMin = std::min(bottomMin, yu * fy + cy);

        // Left edge: all points along x=0
        xd = (0.0f - cx) / fx; yd = (t * h - cy) / fy;
        undistortPoint(xd, yd, k1, k2, p1, p2, k3, xu, yu);
        leftMax = std::max(leftMax, xu * fx + cx);

        // Right edge: all points along x=w-1
        xd = ((float)(w-1) - cx) / fx; yd = (t * h - cy) / fy;
        undistortPoint(xd, yd, k1, k2, p1, p2, k3, xu, yu);
        rightMin = std::min(rightMin, xu * fx + cx);
    }

    // Inner rectangle (alpha=0: no black borders)
    int roiX = std::max(0, (int)std::ceil(leftMax));
    int roiY = std::max(0, (int)std::ceil(topMax));
    int roiW = std::min(w, (int)std::floor(rightMin)) - roiX;
    int roiH = std::min(h, (int)std::floor(bottomMin)) - roiY;
    if (roiW <= 0 || roiH <= 0) { roiX = 0; roiY = 0; roiW = w; roiH = h; }

    // Undistort: for each pixel in the output (undistorted) image,
    // apply forward distortion to find source pixel in distorted input
    Image undist;
    undist.width = w;
    undist.height = h;
    undist.data.resize(w * h * 3);

    for (int oy = 0; oy < h; oy++) {
        for (int ox = 0; ox < w; ox++) {
            float x = ((float)ox - cx) / fx;
            float y = ((float)oy - cy) / fy;
            float xd_n, yd_n;
            distortPoint(x, y, k1, k2, p1, p2, k3, xd_n, yd_n);
            float srcX = xd_n * fx + cx;
            float srcY = yd_n * fy + cy;

            float pixel[3];
            bilinearSample(src, srcX, srcY, pixel);
            float *out = &undist.data[(oy * w + ox) * 3];
            out[0] = pixel[0];
            out[1] = pixel[1];
            out[2] = pixel[2];
        }
    }

    // Crop to ROI
    Image cropped;
    cropped.width = roiW;
    cropped.height = roiH;
    cropped.data.resize(roiW * roiH * 3);
    for (int y = 0; y < roiH; y++) {
        memcpy(&cropped.data[y * roiW * 3],
               &undist.data[((y + roiY) * w + roiX) * 3],
               roiW * 3 * sizeof(float));
    }

    UndistortResult result;
    result.image = std::move(cropped);
    result.fx = fx;
    result.fy = fy;
    result.cx = cx - roiX;
    result.cy = cy - roiY;
    result.width = roiW;
    result.height = roiH;
    return result;
}

UndistortTrainingTargetResult undistortImageAndCoverageMask(
    const Image &image, const CoverageMask &coverageMask,
    float fx, float fy, float cx, float cy,
    float k1, float k2, float p1, float p2, float k3) {
    const size_t maskElements = checkedElementCount(
        coverageMask.width, coverageMask.height, 1, sizeof(uint8_t),
        "Training mask to undistort");
    if (coverageMask.data.size() != maskElements) {
        throw std::invalid_argument(
            "Training mask storage does not match its dimensions");
    }
    if (coverageMask.width != image.width ||
        coverageMask.height != image.height) {
        throw std::invalid_argument(
            "Training mask dimensions do not match the image");
    }

    UndistortResult imageResult = undistortImage(
        image, fx, fy, cx, cy, k1, k2, p1, p2, k3);
    const int roiX = static_cast<int>(std::lround(cx - imageResult.cx));
    const int roiY = static_cast<int>(std::lround(cy - imageResult.cy));

    CoverageMask croppedMask;
    croppedMask.width = imageResult.width;
    croppedMask.height = imageResult.height;
    croppedMask.data.resize(checkedElementCount(
        croppedMask.width, croppedMask.height, 1, sizeof(uint8_t),
        "Undistorted training mask"));
    for (int y = 0; y < croppedMask.height; ++y) {
        for (int x = 0; x < croppedMask.width; ++x) {
            const float undistortedX = static_cast<float>(x + roiX);
            const float undistortedY = static_cast<float>(y + roiY);
            const float normalizedX = (undistortedX - cx) / fx;
            const float normalizedY = (undistortedY - cy) / fy;
            float distortedX = 0.0f;
            float distortedY = 0.0f;
            distortPoint(normalizedX, normalizedY,
                         k1, k2, p1, p2, k3,
                         distortedX, distortedY);
            const float sourceX = distortedX * fx + cx;
            const float sourceY = distortedY * fy + cy;
            croppedMask.data[static_cast<size_t>(y) * croppedMask.width + x] =
                bilinearSampleCoverage(coverageMask, sourceX, sourceY);
        }
    }

    UndistortTrainingTargetResult result;
    result.image = std::move(imageResult.image);
    result.coverageMask = std::move(croppedMask);
    result.fx = imageResult.fx;
    result.fy = imageResult.fy;
    result.cx = imageResult.cx;
    result.cy = imageResult.cy;
    result.width = imageResult.width;
    result.height = imageResult.height;
    return result;
}
