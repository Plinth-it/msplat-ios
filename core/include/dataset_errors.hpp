#ifndef MSPLAT_DATASET_ERRORS_H
#define MSPLAT_DATASET_ERRORS_H

#include <stdexcept>

namespace msplat {

/// The dataset is structurally readable but its declared contents are invalid.
class InvalidDatasetError final : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

/// A dataset asset changed after its metadata was inspected. This remains a
/// runtime error to C++ callers but is still an invalid dataset at the C ABI.
class DatasetChangedError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// A dataset asset could not be opened or decoded, including access denied by
/// a missing or inactive security-scoped resource lease.
class DatasetIOError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

} // namespace msplat

#endif // MSPLAT_DATASET_ERRORS_H
