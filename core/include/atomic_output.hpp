#pragma once

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <unistd.h>

namespace msplat::detail {

/// Owns a uniquely-created sibling temporary file until it is atomically
/// renamed over its destination. The destination is untouched unless commit
/// succeeds, and cleanup can only remove the path created by this instance.
class AtomicOutputFile {
public:
    explicit AtomicOutputFile(const std::string &destination)
        : destinationPath(destination) {
        createTemporary();
    }

    ~AtomicOutputFile() {
        if (!committed) {
            std::error_code ignored;
            std::filesystem::remove(temporaryPath, ignored);
        }
    }

    AtomicOutputFile(const AtomicOutputFile&) = delete;
    AtomicOutputFile& operator=(const AtomicOutputFile&) = delete;

    const std::filesystem::path& temporary() const { return temporaryPath; }

    void commit(const char *formatName) {
        std::error_code error;
        std::filesystem::rename(temporaryPath, destinationPath, error);
        if (error) {
            throw std::runtime_error(
                std::string("Cannot replace ") + formatName + " file: " +
                destinationPath.string() + ": " + error.message());
        }
        committed = true;
    }

private:
    static inline std::atomic<uint64_t> nextSuffix{0};

    [[noreturn]] void throwCreateError(int errorNumber) const {
        throw std::runtime_error(
            "Cannot create temporary output beside " + destinationPath.string() +
            ": " + std::error_code(errorNumber, std::generic_category()).message());
    }

    void createTemporary() {
        constexpr int maxAttempts = 1024;
        for (int attempt = 0; attempt < maxAttempts; ++attempt) {
            temporaryPath = destinationPath.string() + ".tmp." +
                std::to_string(static_cast<long long>(::getpid())) + "." +
                std::to_string(nextSuffix.fetch_add(1, std::memory_order_relaxed));

            const int descriptor = ::open(
                temporaryPath.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
            if (descriptor >= 0) {
                if (::close(descriptor) != 0) {
                    const int closeError = errno;
                    std::error_code ignored;
                    std::filesystem::remove(temporaryPath, ignored);
                    throwCreateError(closeError);
                }
                return;
            }

            const int openError = errno;
            if (openError != EEXIST)
                throwCreateError(openError);
        }

        throw std::runtime_error(
            "Cannot create a unique temporary output beside " +
            destinationPath.string());
    }

    std::filesystem::path destinationPath;
    std::filesystem::path temporaryPath;
    bool committed = false;
};

} // namespace msplat::detail
