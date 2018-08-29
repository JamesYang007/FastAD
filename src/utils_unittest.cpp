#include "utils_unittest.h"

namespace {

    TEST(utils_test, valuetype_test) {
        using namespace ad::core::test;
        bool x;
        x = std::is_same<typename utils::valuetype<
            DualNum<double>, DualNum<double> 
            >, DualNum<double>
            >::value;
        EXPECT_EQ(x, 1);
    }

} // end namespace
