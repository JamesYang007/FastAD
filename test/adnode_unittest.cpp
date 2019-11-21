#include <fastad/adeval.hpp>
#include <fastad/admath.hpp>
#include <fastad/advec.hpp>
#include "gtest/gtest.h"
#include <random>
#include <time.h>


namespace {

TEST(adnode, memory) {
    using namespace ad;
    double x1 = 2.0, x2 = 1.31, x3 = -3.14;
    double dfs[3] = { 0 };
    Var<double> leaf1(x1, dfs);
    Var<double> leaf2(x2, dfs + 1);
    Var<double> leaf3(x3, dfs + 2);

    auto res = leaf1 + sin(leaf2 + leaf3);
    EXPECT_EQ(res.feval(), x1 + std::sin(x2 + x3));
}

TEST(adnode, leaf) {
    using namespace ad::core;
    LeafNode<double> node(0.0, 0, 1.0);
    EXPECT_EQ(node.w == 0.0, 1);
    EXPECT_EQ(node.df == 1.0, 1);
    EXPECT_EQ(node.feval() == 0., 1);
    auto& x = node.self();
    bool b = std::is_same<
        decltype(x)
        , LeafNode<double>&>::value;
    EXPECT_EQ(b, 1);

    LeafNode<double> const node_const(1.0, 0, -1.0);
    EXPECT_EQ(node_const.w == 1.0, 1);
    EXPECT_EQ(node_const.df == -1.0, 1);
    auto& tmp = node_const.self();
    b = std::is_same<
        decltype(tmp)
        , LeafNode<double> const&>::value;
    EXPECT_EQ(b, 1);
}

TEST(adnode, unary) {
    using namespace ad::core;
    double df = 0.;
    LeafNode<double> leaf(0.0, &df, 0);
    UnaryNode<double, ad::math::Sin<double>, LeafNode<double>> unary(leaf);
    // Constructor ok?
    EXPECT_EQ(unary.w == 0.0, 1);
    EXPECT_EQ(unary.df == 0.0, 1);

    // Derived ok?
    auto& x = unary.self();
    bool b = std::is_same<
        decltype(x)
        , UnaryNode<double
        , ad::math::Sin<double>
        , LeafNode<double>
        >&>::value;
    EXPECT_EQ(b, 1);

    UnaryNode<double, ad::math::Sin<double>, LeafNode<double>> const unary_const(leaf);
    EXPECT_EQ(unary_const.w == 0.0, 1);
    EXPECT_EQ(unary_const.df == 0.0, 1);
    auto& tmp = unary_const.self();
    b = std::is_same<
        decltype(tmp)
        , UnaryNode<double
        , ad::math::Sin<double>
        , LeafNode<double>
        >const&>::value;
    EXPECT_EQ(b, 1);

    // feval
    EXPECT_EQ(unary.feval() == 0.0, 1);

    // beval
    // dsin/dx(0) = cos(0) = 1
    unary.beval(1);
    EXPECT_EQ(df == 1, 1);
    EXPECT_EQ(unary.lhs.df == 1, 1);
}


// More complicated example
// LeafNode -> UnaryNode -> UnaryNode
TEST(adnode, unary_complex) {
    using namespace ad;
    double df = 0.;
    core::LeafNode<double> leaf(3.14, &df, 0.0);
    auto sin_unary = sin(leaf);
    auto sin_sin_unary = sin(sin_unary);
    double fx = sin_sin_unary.feval();
    EXPECT_EQ(fx, std::sin(std::sin(3.14)));
    sin_sin_unary.beval(1);
    EXPECT_EQ(df, std::cos(std::sin(3.14)) * std::cos(3.14));
    // This is because sin_unary is copied
    EXPECT_EQ(sin_sin_unary.lhs.df, std::cos(std::sin(3.14)));
    //EXPECT_EQ(sin_unary.df, std::cos(std::sin(3.14))); 
    EXPECT_EQ(sin_unary.df, 0);
}


// BinaryNode
TEST(adnode, binary) {
    using namespace ad;
    double df1 = 0., df2 = 0.;
    core::LeafNode<double> leaf1(1.0, &df1, 0.0);
    core::LeafNode<double> leaf2(0.0, &df2, 0.0);

    auto binary = leaf1 + leaf2;
    // feval
    EXPECT_EQ(binary.lhs.w, 1.0);
    EXPECT_EQ(binary.rhs.w, 0.0);
    EXPECT_EQ(binary.feval(), 1.0);

    // beval
    binary.beval(1);
    EXPECT_EQ(df1, 1.0);
    EXPECT_EQ(df2, 1.0);
}

// BinaryNode more complex
TEST(adnode, binary_complex) {
    using namespace ad;
    double df1 = 0., df2 = 0., df3 = 0., df4 = 0.;
    core::LeafNode<double> leaf1(1.0, &df1, 0.0);
    core::LeafNode<double> leaf2(0.0, &df2, 0.0);
    core::LeafNode<double> leaf3(2.0, &df3, 0.0);
    core::LeafNode<double> leaf4(-3.0, &df4, 0.0);

    auto&& binary = leaf1 + leaf2 + leaf3 + leaf4;
    // feval
    EXPECT_EQ(binary.feval(), 0.0);
    EXPECT_EQ(binary.lhs.w, 3.0);

    // beval
    binary.beval(1);
    EXPECT_EQ(df1, 1.0);
    EXPECT_EQ(df2, 1.0);
    EXPECT_EQ(df3, 1.0);
    EXPECT_EQ(df4, 1.0);
}

// Glueing style
TEST(adnode, glue) {
    using namespace ad::core;
    LeafNode<double> w1(1.0);
    LeafNode<double> w2(2.0);
    LeafNode<double> w3(3.0);
    LeafNode<double> w4(4.0);

    auto subexpr = w1 * w2;
    EXPECT_EQ(subexpr.lhs.w, 1.0);
    EXPECT_EQ(subexpr.lhs.w_ptr, w1.w_ptr);
    EXPECT_EQ(subexpr.rhs.w, 2.0);
    EXPECT_EQ(subexpr.rhs.w_ptr, w2.w_ptr);

    auto subtree = (w3 = w1 * w2);
    EXPECT_EQ(subtree.lhs.w, 3.0);
    EXPECT_EQ(subtree.lhs.w_ptr, w3.w_ptr);
    EXPECT_EQ(subtree.rhs.lhs.w, 1.0);
    EXPECT_EQ(subtree.rhs.rhs.w, 2.0);
    EXPECT_EQ(subtree.rhs.lhs.w_ptr, w1.w_ptr);
    EXPECT_EQ(subtree.rhs.rhs.w_ptr, w2.w_ptr);

    auto subtree2 = (w4 = w3 * w3);
    EXPECT_EQ(subtree2.lhs.w, 4.0);
    EXPECT_EQ(subtree2.lhs.w_ptr, w4.w_ptr);
    EXPECT_EQ(subtree2.rhs.lhs.w, 3.0);
    EXPECT_EQ(subtree2.rhs.rhs.w, 3.0);
    EXPECT_EQ(subtree2.rhs.lhs.w_ptr, w3.w_ptr);
    EXPECT_EQ(subtree2.rhs.rhs.w_ptr, w3.w_ptr);

    //auto expr = (subtree, subtree2);
    auto expr = (
        w3 = w1 * w2
        , w4 = w3 * w3
        //w4 = w1*w2*w1*w2
        );
    //EXPECT_EQ(expr.lhs.lhs.w, 3.0);

    ad::Evaluate(expr);
    EXPECT_EQ(w4.w, 4.0);
    EXPECT_EQ(w3.w, 2.0);
    //EXPECT_EQ(w3.w, 3.0);
    EXPECT_EQ(w2.w, 2.0);
    EXPECT_EQ(w1.w, 1.0);

    EXPECT_NE(w4.df_ptr, nullptr);
    EXPECT_NE(w3.df_ptr, nullptr);
    EXPECT_NE(w2.df_ptr, nullptr);
    EXPECT_NE(w1.df_ptr, nullptr);

    ad::EvaluateAdj(expr);
    EXPECT_EQ(w4.df, 1.0);
    EXPECT_EQ(w3.df, 2 * w3.w);
    //EXPECT_EQ(w3.df, 0);
    EXPECT_EQ(w2.df, 2 * w2.w*w1.w*w1.w);
    EXPECT_EQ(w1.df, 2 * w1.w*w2.w*w2.w);
}

// Use-case
TEST(adnode, use_case) {
    using namespace ad;
    double x1 = -0.201, x2 = 1.2241;
    Var<double> w1(x1);
    Var<double> w2(x2);

    Var<double> w3;
    Var<double> w4;
    Var<double> w5;
    auto expr = (
        w3 = w1 * sin(w2)
        , w4 = w3 + w1 * w2
        , w5 = exp(w4*w3)
        );
    autodiff(expr);
    //Evaluate(expr);

    EXPECT_EQ(w5.w, std::exp((x1*std::sin(x2) + x1 * x2)*(x1*std::sin(x2))));
    EXPECT_EQ(w4.w, x1*std::sin(x2) + x1 * x2);
    EXPECT_EQ(w3.w, x1*std::sin(x2));

    //EvaluateAdj(expr);
    EXPECT_EQ(w5.df, 1);
    EXPECT_EQ(w4.df, w3.w * w5.w);
    EXPECT_EQ(w3.df, (w3.w + w4.w) * w5.w);
    EXPECT_EQ(w2.df, w5.w*x1*x1*(std::cos(x2)*(std::sin(x2) + x2) + std::sin(x2)*(1 + std::cos(x2))));
    EXPECT_EQ(w1.df, w5.w * 2 * x1 * std::sin(x2) *(std::sin(x2) + x2));
}

//====================================================================================================
// Assumes advec works

    //TEST(adnode, sumnode_foreach) {
    //    using namespace ad;

    //    Vec<double> vec;
    //    std::default_random_engine gen;
    //    std::normal_distribution<double> dist(0.0,1.0);
    //    for (size_t i = 0; i < 1e6; ++i) {
    //        vec.emplace_back(dist(gen));        
    //    }

    //    Vec<double> sumvec(vec.size());
    //    sumvec[0] = vec[0];

    //    Var<double> w4;
    //    Var<double> w5;

    //    auto&& expr = ad::for_each(
    //            boost::counting_iterator<size_t>(1)
    //            , boost::counting_iterator<size_t>(vec.size())
    //            , [&sumvec, &vec](size_t i) 
    //            {return sumvec[i] = sumvec[i-1] + vec[i];}
    //            );

    //    std::clock_t time;
    //    time = std::clock();
    //    autodiff((w4=expr, w5 = w4*w4 + vec[0]));
    //    std::cout << "Autodiff only: " 
    //        << 1e3 * (std::clock() - time) / (double) CLOCKS_PER_SEC
    //        << " ms" << std::endl;

    //    double sqsum = 0.0;
    //    for (size_t i = 0; i < vec.size(); ++i) {
    //        sqsum += vec[i].w;
    //    }
    //    EXPECT_DOUBLE_EQ(w5.w, sqsum*sqsum + vec[0].w);

    //}

TEST(adnode, sumnode) {
    using namespace ad;
    auto init = { 0.203104, 1.4231, -1.231 };
    Vec<double> vec(init);
    auto expr = sum(vec.vec.begin(), vec.vec.end(),
        [](Var<double> const& v) {return ad::cos(ad::sin(v)*v); });
    EXPECT_EQ(expr.w, 0);
    EXPECT_EQ(expr.df, 0);
    EXPECT_EQ(expr.start, vec.vec.begin());
    EXPECT_EQ(expr.end, vec.vec.end());

    double actual_sum = 0;
    for (size_t i = 0; i < vec.size(); ++i)
        actual_sum += std::cos(std::sin(vec[i].w)*vec[i].w);

    EXPECT_DOUBLE_EQ(expr.feval(), actual_sum);
    expr.beval(1);
    for (size_t i = 0; i < vec.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec[i].df,
            -std::sin(std::sin(vec[i].w)*vec[i].w)*(std::cos(vec[i].w)*vec[i].w + std::sin(vec[i].w)));
    }
    EXPECT_DOUBLE_EQ(expr.w, actual_sum);
}

// More complex example
TEST(adnode, sumnode_complex) {
    using namespace ad;
    //auto init = {0.203104, 1.4231, -1.231};
    //Vec<double> vec(init);
    std::default_random_engine gen;
    std::normal_distribution<double> dist(0.0, 1.0);
    Vec<double> vec;
    for (size_t i = 0; i < 1e4; ++i) {
        vec.emplace_back(dist(gen));
    }

    Var<double> w4;
    Var<double> w5;
    auto expr = sum(vec.begin(), vec.end(),
        [](Var<double> const& v) {return v * v; });
    std::clock_t time;
    time = std::clock();
    autodiff((w4 = expr, w5 = w4 * w4 + vec[0]));
    std::cout << "Autodiff only: "
        << 1e3 * (std::clock() - time) / (double)CLOCKS_PER_SEC
        << " ms" << std::endl;

    double sqsum = 0.0;
    for (size_t i = 0; i < vec.size(); ++i) {
        sqsum += vec[i].w*vec[i].w;
    }
    EXPECT_DOUBLE_EQ(w5.w, sqsum*sqsum + vec[0].w);

    for (size_t i = 0; i < vec.size(); ++i) {
        double correction = (i == 0) ? 1 : 0;
        EXPECT_DOUBLE_EQ(vec[i].df, correction + 4 * w4.w*vec[i].w);
    }

}

// Even more complex
TEST(adnode, sumnode_complex_2) {
    using namespace ad;
    std::default_random_engine gen;
    std::normal_distribution<double> dist(0.0, 1.0);
    Vec<double> vec;
    for (size_t i = 0; i < 1e6; ++i) {
        vec.emplace_back(dist(gen));
    }

    Var<double> w4;
    Var<double> w5;
    auto expr = sum(vec.begin(), vec.end(),
        [](Var<double> const& v) {return v * v; });
    std::clock_t time;
    time = std::clock();
    autodiff((w4 = expr, w5 = w4 * w4 + vec[0]));
    std::cout << "Autodiff only: "
        << 1e3 * (std::clock() - time) / (double)CLOCKS_PER_SEC
        << " ms" << std::endl;

    double sqsum = 0.0;
    for (size_t i = 0; i < vec.size(); ++i) {
        sqsum += vec[i].w*vec[i].w;
    }
    EXPECT_DOUBLE_EQ(w5.w, sqsum*sqsum + vec[0].w);

    for (size_t i = 0; i < vec.size(); ++i) {
        double correction = (i == 0) ? 1 : 0;
        EXPECT_DOUBLE_EQ(vec[i].df, correction + 4 * w4.w*vec[i].w);
    }

}

// ForEach
TEST(adnode, foreach) {
    using namespace ad;
    auto init = { 100., 20., -10. };
    Vec<double> vec(init);
    vec.emplace_back(1e-3);
    Vec<double> prod(4);
    prod[0] = vec[0];
    auto&& expr = ad::for_each(
        boost::counting_iterator<size_t>(1)
        , boost::counting_iterator<size_t>(4)
        , [&vec, &prod](size_t i) {return prod[i] = prod[i - 1] * vec[i]; }
    );

    double actual = 1;
    for (size_t i = 0; i < vec.size(); ++i)
        actual *= vec[i].w;

    Var<double> res;
    Var<double> w4;
    autodiff((res = expr, w4 = res * res + vec[0]));
    EXPECT_DOUBLE_EQ(res.w, actual);
    EXPECT_DOUBLE_EQ(w4.w, actual*actual + vec[0].w);
    for (size_t i = 0; i < vec.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec[i].df,
            ((i == 0) ? 1 : 0) + 2 * actual * actual / vec[i].w);
    }
}

// TODO: move this to benchmark
// ForEach complex
//TEST(adnode, foreach_complex) {
//    using namespace ad;
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::normal_distribution<long double> dist(0,1);
//    std::uniform_real_distribution<long double> udist(-1e-8, 1e-8);

//    for (size_t trial=0; trial < 1e1; ++trial) {

//    Vec<long double> vec(0); // set capacity
//    vec.emplace_back(udist(gen));
//    //vec.emplace_back(0);
//    for (size_t i = 0; i < 1e1; ++i) {
//        vec.emplace_back(dist(gen));        
//    }

//    Var<long double> w4;
//    Var<long double> w5;
//    Vec<long double> prod(vec.size());
//    prod[0] = vec[0];
//    auto&& expr = ad::for_each(
//            boost::counting_iterator<size_t>(1)
//            , boost::counting_iterator<size_t>(vec.size())
//            , [&vec, &prod](size_t i) {return prod[i] = prod[i-1] * vec[i];});
//    //auto expr = ad::for_each<long double>(vec.begin(), vec.end(), 
//    //        [](Var<long double> const& v) {return v*v;});
//    autodiff((w4=expr, w5 = w4*w4 + vec[0]));

//    long double total = 1.0;
//    for (size_t i = 0; i < vec.size(); ++i) {
//        total *= vec[i].w;
//    }
//    EXPECT_DOUBLE_EQ(w5.w, total*total + vec[0].w);
//    //std::cout << "Product of f(w_i): " << w4.w << std::endl;

//    for (size_t i = 0; i < vec.size(); ++i) {
//        long double correction = (i==0) ? 1:0;
//        long double factor = (vec[i].w == 0) ? 0:2*w4.w*w4.w/vec[i].w;
//        EXPECT_DOUBLE_EQ(vec[i].df, correction + factor);
//    }
//    }
//}

// ProdNode
TEST(adnode, prodnode) {
    using namespace ad;
    auto init = { 100., 20., -10. };
    Vec<double> vec(init);
    vec.emplace_back(1e-3);
    auto&& expr = ad::prod(
        boost::counting_iterator<size_t>(0)
        , boost::counting_iterator<size_t>(4)
        , [&vec](size_t i) {return vec[i] * vec[i]; });

    double actual = 1;
    for (size_t i = 0; i < vec.size(); ++i)
        actual *= vec[i].w * vec[i].w;

    Var<double> res;
    Var<double> w4;
    autodiff((res = expr, w4 = res * res + vec[0]));
    EXPECT_DOUBLE_EQ(res.w, actual);
    EXPECT_DOUBLE_EQ(w4.w, actual*actual + vec[0].w);
    for (size_t i = 0; i < vec.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec[i].df,
            ((i == 0) ? 1 : 0) + 4 * actual * actual / vec[i].w);
    }
}

// ProdNode complex
TEST(adnode, prodnode_complex) {
    using namespace ad;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<long double> dist(0, 1);
    std::uniform_real_distribution<long double> udist(-1e-8, 1e-8);
    std::clock_t time;

    Vec<long double> vec; // set capacity
    vec.emplace_back(udist(gen));
    for (size_t i = 0; i < 1e6 - 1; ++i) {
        vec.emplace_back(dist(gen));
    }

    Var<long double> w4;
    Var<long double> w5;
    auto&& expr = ad::prod(
        boost::counting_iterator<size_t>(0)
        , boost::counting_iterator<size_t>(vec.size())
        , [&vec](size_t i) {return vec[i] * vec[i]; });

    time = std::clock();
    autodiff((w4 = expr, w5 = w4 * w4 + vec[0]));
    std::cout << "Autodiff only: "
        << 1e3 * (std::clock() - time) / (double)CLOCKS_PER_SEC
        << " ms" << std::endl;

    long double total = 1.0;
    for (size_t i = 0; i < vec.size(); ++i) {
        total *= vec[i].w * vec[i].w;
    }
    EXPECT_DOUBLE_EQ(w5.w, total*total + vec[0].w);

    for (size_t i = 0; i < vec.size(); ++i) {
        long double correction = (i == 0) ? 1 : 0;
        long double factor = (vec[i].w == 0) ? 0 : 4 * w4.w*w4.w / vec[i].w;
        EXPECT_DOUBLE_EQ(vec[i].df, correction + factor);
    }
}

} // namespace
