/*#########################################################################

 Median helpers, copied unmodified from tvpham/iq (src/utils.h), which is
 already pure C++ with no R dependency.

 Original author: Thang V. Pham, t.pham@amsterdamumc.nl
 Source: https://github.com/tvpham/iq  (BSD 3-Clause)

#########################################################################*/

#ifndef __IQ_UTILS_H
#define __IQ_UTILS_H

#include <algorithm>
#include <cstddef>

inline double quick_median(double *median_buffer, std::size_t n) {

    const std::size_t mid = n / 2;

    std::nth_element(median_buffer, median_buffer + mid, median_buffer + n);

    if (n % 2 == 1) {
        return median_buffer[mid];
    } else {
        double mid_val = median_buffer[mid];
        std::nth_element(median_buffer, median_buffer + mid - 1, median_buffer + n);
        return (median_buffer[mid - 1] + mid_val) / 2.0;
    }
}

#endif  // __IQ_UTILS_H
