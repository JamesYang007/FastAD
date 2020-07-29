#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/core/bind.hpp>
#include <fastad_bits/reverse/core/var.hpp>
#include <fastad_bits/reverse/core/math.hpp>
#include <fastad_bits/reverse/core/eq.hpp>
#include <fastad_bits/reverse/core/glue.hpp>
#include <fastad_bits/reverse/core/eval.hpp>

namespace ad {

struct bind_fixture : base_fixture
{
protected:
    Var<value_t> w1{1.0}, w2{2.0}, w3{3.0}, w4{4.0};

    auto make_expr_bind() 
    {
        auto expr = (w3 = w1 * w2, w4 = w3 * w3);
        return ad::bind(expr);
    }

    template <class ExprBindType>
    void test(ExprBindType&& expr_bind)
    {
        value_t res = ad::autodiff(expr_bind);
        EXPECT_DOUBLE_EQ(res, 4.);
        EXPECT_DOUBLE_EQ(w1.get(), 1.);
        EXPECT_DOUBLE_EQ(w2.get(), 2.);
        EXPECT_DOUBLE_EQ(w3.get(), 2.);
        EXPECT_DOUBLE_EQ(w4.get(), 4.);

        EXPECT_DOUBLE_EQ(w4.get_adj(0,0), 1.0);
        EXPECT_DOUBLE_EQ(w3.get_adj(0,0), 2 * w3.get());
        EXPECT_DOUBLE_EQ(w2.get_adj(0,0), 2 * w2.get()*w1.get()*w1.get());
        EXPECT_DOUBLE_EQ(w1.get_adj(0,0), 2 * w1.get()*w2.get()*w2.get());
    }
};

TEST_F(bind_fixture, bind_test_lref) 
{
    auto expr_bind = make_expr_bind();
    test(expr_bind);
    w1.reset_adj();
    w2.reset_adj();
    w3.reset_adj();
    w4.reset_adj();
    // second time should not need any change in cache
    // only requires reseting adjoints of wi's
    test(expr_bind);   
}

TEST_F(bind_fixture, bind_test_rref) 
{
    test(make_expr_bind());
    w1.reset_adj();
    w2.reset_adj();
    w3.reset_adj();
    w4.reset_adj();
    test(make_expr_bind());
}

} // namespace ad
