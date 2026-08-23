#ifndef METAL_TENSOR_H
#define METAL_TENSOR_H

#include <vector>
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <CoreFoundation/CoreFoundation.h>
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif

// Forward-declare the Metal buffer type for C++ compatibility.
// Full Metal/Metal.h is only needed in .mm files.
#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* MTLBufferRef;  // opaque handle in pure C++
#endif

enum class DType : uint8_t {
    Float32,
    Int32,
    Int64,
    UInt8,
    Float64,
};

inline size_t dtypeSize(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
        case DType::UInt8:   return 1;
        case DType::Float64: return 8;
    }
}

// Lightweight GPU tensor — wraps an MTLBuffer with shape metadata.
class MTensor {
public:
    MTensor() = default;
    ~MTensor() { reset(); }

    MTensor(const MTensor& other) {
        copyFrom(other);
    }

    MTensor& operator=(const MTensor& other) {
        if (this != &other) {
            reset();
            copyFrom(other);
        }
        return *this;
    }

    MTensor(MTensor&& other) noexcept {
        moveFrom(std::move(other));
    }

    MTensor& operator=(MTensor&& other) noexcept {
        if (this != &other) {
            reset();
            moveFrom(std::move(other));
        }
        return *this;
    }

#ifdef __OBJC__
    // GPU allocation (Objective-C++ only)
    MTensor(id<MTLDevice> device, std::vector<int64_t> shape, DType dtype)
        : _shape(std::move(shape)), _dtype(dtype) {
        _numel = 1;
        for (auto s : _shape) _numel *= s;
        size_t bytes = _numel * dtypeSize(_dtype);
        if (bytes == 0) bytes = 4;
        id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf) throwAllocationFailure(bytes);
#if __has_feature(objc_arc)
        _buffer = (__bridge_retained void*)buf;
#else
        _buffer = (void*)buf;
#endif
        _ownsBuffer = true;
        _data = [buf contents];  // cache CPU-accessible pointer for C++ access
    }

    id<MTLBuffer> buffer() const { return (__bridge id<MTLBuffer>)_buffer; }
#endif

    // CPU allocation (no Metal buffer)
    MTensor(std::vector<int64_t> shape, DType dtype)
        : _shape(std::move(shape)), _dtype(dtype) {
        _numel = 1;
        for (auto s : _shape) _numel *= s;
        size_t bytes = _numel * dtypeSize(_dtype);
        _cpu_data.resize(bytes);
    }

    bool defined() const { return _buffer != nullptr || !_cpu_data.empty(); }
    bool isGpu() const { return _buffer != nullptr; }

    int64_t numel() const { return _numel; }
    int64_t size(int dim) const {
        if (dim < 0) dim += _shape.size();
        return _shape[dim];
    }
    int ndim() const { return (int)_shape.size(); }
    const std::vector<int64_t>& shape() const { return _shape; }
    DType dtype() const { return _dtype; }
    size_t elementSize() const { return dtypeSize(_dtype); }
    size_t nbytes() const { return _numel * dtypeSize(_dtype); }

    void* data_ptr() {
        if (_data) return _data;
        return _cpu_data.data();
    }
    const void* data_ptr() const {
        if (_data) return _data;
        return _cpu_data.data();
    }

    template<typename T> T* data() { return static_cast<T*>(data_ptr()); }
    template<typename T> const T* data() const { return static_cast<const T*>(data_ptr()); }

    void zero() {
        memset(data_ptr(), 0, _numel * dtypeSize(_dtype));
    }

    // Create a CPU copy of the data
    MTensor cpu() const {
        MTensor out(_shape, _dtype);
        memcpy(out.data_ptr(), data_ptr(), nbytes());
        return out;
    }

    void reset() {
        if (_buffer && _ownsBuffer) CFRelease(_buffer);
        _buffer = nullptr;
        _data = nullptr;
        _ownsBuffer = false;
        _cpu_data.clear();
        _shape.clear();
        _numel = 0;
    }

    // Stride for dim 0 (elements per row)
    int64_t stride0() const {
        if (_shape.size() <= 1) return 1;
        int64_t s = 1;
        for (size_t i = 1; i < _shape.size(); i++) s *= _shape[i];
        return s;
    }

    // Create a view of the first `n` elements along dim 0.
    // WARNING: Non-owning — shares the underlying MTLBuffer without retaining it.
    // The caller MUST ensure the parent MTensor outlives all views.
    // Use-after-free if the parent is destroyed while a view exists.
    MTensor view(int64_t n) const {
        MTensor v;
        v._buffer = _buffer;  // shares the buffer (non-owning)
        v._data = _data;      // shares the CPU-accessible pointer
        v._ownsBuffer = false;
        v._shape = _shape;
        v._shape[0] = n;
        v._dtype = _dtype;
        v._numel = n * stride0();
        return v;
    }

private:
    // Metal returns nil rather than raising when a buffer allocation cannot be
    // satisfied. Callers here treat the result as valid, so an unchecked nil
    // reaches a compute encoder and the kernel faults dereferencing null — the
    // failure then surfaces as GPU corruption rather than as an allocation
    // error, and takes the process's GPU context down with it. Fail here, while
    // we still know what was being asked for.
    [[noreturn]] void throwAllocationFailure(size_t bytes) const {
        char shapeText[128] = "scalar";
        int n = 0;
        for (size_t i = 0; i < _shape.size() && n < (int)sizeof(shapeText) - 24; i++) {
            n += snprintf(shapeText + n, sizeof(shapeText) - n, "%s%lld",
                          i ? "x" : "", (long long)_shape[i]);
        }

        char message[320];
#if TARGET_OS_IPHONE
        snprintf(message, sizeof(message),
                 "MTLBuffer allocation failed: [%s] %.1f MB requested, "
                 "%.1f MB available before jetsam",
                 shapeText, (double)bytes / (1024.0 * 1024.0),
                 (double)os_proc_available_memory() / (1024.0 * 1024.0));
#else
        snprintf(message, sizeof(message),
                 "MTLBuffer allocation failed: [%s] %.1f MB requested",
                 shapeText, (double)bytes / (1024.0 * 1024.0));
#endif
        // Keep the diagnostic for native/CLI callers; the checked C ABI also
        // translates this exception into MSPLAT_STATUS_OUT_OF_MEMORY.
        fprintf(stderr, "MSPLAT_ALLOC_FAIL %s\n", message);
        fflush(stderr);
        throw std::runtime_error(message);
    }

    void copyFrom(const MTensor& other) {
        _shape = other._shape;
        _dtype = other._dtype;
        _numel = other._numel;
        _data = other._data;
        _buffer = other._buffer;
        _cpu_data = other._cpu_data;
        _ownsBuffer = false;
        if (_buffer) {
            CFRetain(_buffer);
            _ownsBuffer = true;
        }
    }

    void moveFrom(MTensor&& other) {
        _buffer = other._buffer;
        _data = other._data;
        _cpu_data = std::move(other._cpu_data);
        _shape = std::move(other._shape);
        _dtype = other._dtype;
        _numel = other._numel;
        _ownsBuffer = other._ownsBuffer;

        other._buffer = nullptr;
        other._data = nullptr;
        other._shape.clear();
        other._numel = 0;
        other._ownsBuffer = false;
    }

    void* _buffer = nullptr;  // retained id<MTLBuffer> as void*
    void* _data = nullptr;    // cached CPU-accessible pointer (shared memory on Apple Silicon)
    bool _ownsBuffer = false;
    std::vector<uint8_t> _cpu_data;
    std::vector<int64_t> _shape;
    DType _dtype = DType::Float32;
    int64_t _numel = 0;
};

// Factory helpers (Objective-C++ only)
#ifdef __OBJC__
inline MTensor mtensor_empty(id<MTLDevice> dev, std::vector<int64_t> shape, DType dt) {
    return MTensor(dev, std::move(shape), dt);
}

inline MTensor mtensor_zeros(id<MTLDevice> dev, std::vector<int64_t> shape, DType dt) {
    MTensor t(dev, std::move(shape), dt);
    t.zero();
    return t;
}
#endif

#endif
