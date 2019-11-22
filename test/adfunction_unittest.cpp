#include <fastad_bits/adeval.hpp>
#include <fastad_bits/admath.hpp>
#include <fastad_bits/adfunction.hpp>
#include "gtest/gtest.h"

namespace {

auto F_lmda = MAKE_LMDA(ad::sin(x[0]));
auto G_lmda = MAKE_LMDA(ad::cos(x[0]));

// Scalar
TEST(adfunction, scalar_copy_ctor) {
    using namespace ad;
    double x[] = { 1. };
    auto F = make_function(F_lmda);
    auto G = F; // copy construct

    // Check that x,w are independent for F,G
    autodiff(G(x, x + 1));
    EXPECT_EQ(F.x.size(), 0);
    EXPECT_EQ(F.w.size(), 0);
    EXPECT_EQ(G.x.size(), 1);
    EXPECT_EQ(G.w.size(), 1);

    // Check that autodiff results are the same
    autodiff(F(x, x + 1));
    EXPECT_DOUBLE_EQ(F.x[0].w, G.x[0].w);
    EXPECT_DOUBLE_EQ(F.x[0].df, G.x[0].df);
}

// Vector
TEST(adfunction, vector_copy_ctor) {
    using namespace ad;
    double x[] = { 1. };
    auto F = make_function(F_lmda);
    auto G = make_function(G_lmda);
    auto FG = make_function(F, G);

    EXPECT_NE(&(std::get<0>(FG.tup)), &F);
    EXPECT_NE(&(std::get<1>(FG.tup)), &G);

    auto&& expr = FG(x, x + 1);
    autodiff(expr);

    EXPECT_EQ(F.x.size(), 0);
    EXPECT_EQ(F.w.size(), 0);
    EXPECT_EQ(G.x.size(), 0);
    EXPECT_EQ(G.w.size(), 0);
    EXPECT_EQ(std::get<0>(FG.tup).x.size(), 1);
    EXPECT_EQ(std::get<0>(FG.tup).w.size(), 1);
    EXPECT_EQ(std::get<1>(FG.tup).x.size(), 1);
    EXPECT_EQ(std::get<1>(FG.tup).w.size(), 1);

    EXPECT_NE(std::get<0>(FG.tup).x.size(), F.x.size());
    EXPECT_NE(std::get<1>(FG.tup).x.size(), G.x.size());
    EXPECT_NE(std::get<0>(FG.tup).w.size(), F.w.size());
    EXPECT_NE(std::get<1>(FG.tup).w.size(), G.w.size());

}

// ComposedFunction (scalar)
TEST(adfunction, composed_scalar_ctor) {
    using namespace ad;
    double x[] = { 1. };
    auto F = make_function(F_lmda);
    auto G = make_function(G_lmda);
    auto F_G = compose(F, G);
    autodiff(F_G(x, x + 1));

    EXPECT_NE(&F_G.composer, &F);
    EXPECT_NE(&F_G.composed, &G);

    EXPECT_EQ(F.x.size(), 0);
    EXPECT_EQ(F.w.size(), 0);
    EXPECT_EQ(G.x.size(), 0);
    EXPECT_EQ(G.w.size(), 0);

    EXPECT_EQ(F_G.composer.w.size(), 1);
    EXPECT_EQ(F_G.composed.w.size(), 1);
    EXPECT_EQ(F_G.x.size(), 1);
    EXPECT_EQ(F_G.w.size(), 1);
}

// ComposedFunction (vector)
TEST(adfunction, composed_vector_ctor) {
    using namespace ad;
    double x[] = { 0.2, 1.59 };
    auto F_lmda = MAKE_LMDA(
        ad::sin(x[0]) + x[1],
        w[0] * x[1]
    );

    auto G_lmda = MAKE_LMDA(
        ad::cos(x[0])
    );

    auto H_lmda = MAKE_LMDA(
        ad::log(x[0])
    );
    auto FG = make_function(F_lmda, G_lmda);
    auto GH = make_function(G_lmda, H_lmda);
    auto FG_GH = compose(FG, GH);
    auto&& expr = FG_GH(x, x + 2);
    autodiff(expr);

    EXPECT_NE(&(std::get<0>(FG_GH.tup).composed), &GH);
    EXPECT_NE(&(std::get<1>(FG_GH.tup).composed), &GH);
    EXPECT_NE(&(std::get<0>(FG_GH.tup).composer), &(std::get<0>(FG.tup)));
    EXPECT_NE(&(std::get<1>(FG_GH.tup).composer), &(std::get<1>(FG.tup)));

    EXPECT_EQ(std::get<0>(FG.tup).x.size(), 0);
    EXPECT_EQ(std::get<1>(FG.tup).x.size(), 0);
    EXPECT_EQ(std::get<0>(GH.tup).x.size(), 0);
    EXPECT_EQ(std::get<1>(GH.tup).x.size(), 0);

    EXPECT_EQ(std::get<0>(FG.tup).w.size(), 0);
    EXPECT_EQ(std::get<1>(FG.tup).w.size(), 0);
    EXPECT_EQ(std::get<0>(GH.tup).w.size(), 0);
    EXPECT_EQ(std::get<1>(GH.tup).w.size(), 0);
}

} // end namespace 
