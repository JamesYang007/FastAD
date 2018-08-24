#pragma once
#include <algorithm>

namespace utils {

// Returns data type
// Ensures TL and TR have same datatype
template <class TL, class TR>
using valuetype = typename std::enable_if<
                std::is_same<
                    TL, TR
                    >::value
                , TL
                >::type;

} // end utils
