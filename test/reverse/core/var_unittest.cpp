#include <gtest/gtest.h>
#include <fastad_bits/reverse/core/var.hpp>

namespace ad {
namespace core {

struct var_fixture : ::testing::Test
{
protected:
    using value_t = double;
    using scl_v_t = Var<value_t, scl>;
    using vec_v_t = Var<value_t, vec>;
    using mat_v_t = Var<value_t, mat>;

    template <class T>
    void test_ctor(T&& tmp)
    {
        T x(std::forward<T>(tmp)); 
        T y = x;
        y = x;
        x = std::move(y);
    }
};

TEST_F(var_fixture, var_ctors)
{
    test_ctor(scl_v_t());
    test_ctor(vec_v_t(1));
    test_ctor(mat_v_t(1,2));
}

} // namespace core
} // namespace ad
