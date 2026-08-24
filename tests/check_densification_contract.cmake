if(NOT DEFINED MSPLAT_SOURCE_DIR)
    message(FATAL_ERROR "MSPLAT_SOURCE_DIR is required")
endif()

file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.metal" metal_source)
file(READ "${MSPLAT_SOURCE_DIR}/core/metal/msplat_metal.mm" host_source)

function(require_contains contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Densification contract missing: ${label}")
    endif()
endfunction()

function(require_absent contents needle label)
    string(FIND "${contents}" "${needle}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "Stale densification contract remains: ${label}")
    endif()
endfunction()

function(extract_section source_name start_marker end_marker output_name)
    string(FIND "${${source_name}}" "${start_marker}" section_start)
    if(section_start EQUAL -1)
        message(FATAL_ERROR
            "Densification contract section start missing: ${start_marker}")
    endif()
    string(SUBSTRING "${${source_name}}" ${section_start} -1 section_tail)
    string(FIND "${section_tail}" "${end_marker}" section_length)
    if(section_length EQUAL -1 OR section_length EQUAL 0)
        message(FATAL_ERROR
            "Densification contract section end missing: ${end_marker}")
    endif()
    string(SUBSTRING "${section_tail}" 0 ${section_length}
        section_contents)
    set(${output_name} "${section_contents}" PARENT_SCOPE)
endfunction()

extract_section(metal_source "kernel void densify_classify_kernel("
    "kernel void densify_append_split_kernel(" classifier)

require_contains("${classifier}"
    "bool do_dup = !do_split && high_grad;"
    "mutually exclusive split and duplicate outcomes")
require_absent("${classifier}"
    "bool do_dup = !is_large && high_grad;"
    "world-space-only duplicate predicate")
require_contains("${host_source}"
    "throw std::runtime_error(\"Densification classified one Gaussian twice\")"
    "host overlap invariant")
