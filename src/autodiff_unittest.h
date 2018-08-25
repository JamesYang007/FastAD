#pragma once
#include "autodiff.h"
#include "gtest/gtest.h"

namespace {

    TEST(autodiff_test, constructor) {
        using namespace ad::core;
        double x=1.0, y=2.0;
        auto autodiff = make_AutoDiff(x, y);
        EXPECT_EQ(autodiff.duals[0].w == x, 1);
        EXPECT_EQ(autodiff.duals[1].w == y, 1);
        EXPECT_EQ(autodiff.duals[0].df == 0, 1);
        EXPECT_EQ(autodiff.duals[1].df == 0, 1);
    }
}
