#include "gtest/gtest.h"
#include <fastad_bits/type_traits.hpp>

namespace ad {

namespace util {

struct type_traits_fixture: ::testing::Test 
{
protected:
};

TEST_F(type_traits_fixture, is_pointer_like_pointer)
{
    EXPECT_TRUE(is_pointer_like_dereferenceable<char*>::value);
}

TEST_F(type_traits_fixture, is_pointer_like_iterator)
{
    using iter_t = std::vector<int>::const_iterator;
    EXPECT_TRUE(is_pointer_like_dereferenceable<iter_t>::value);
}

TEST_F(type_traits_fixture, is_tuple_true)
{
    EXPECT_TRUE((is_tuple<std::tuple<int, char>>::value));
}

TEST_F(type_traits_fixture, is_tuple_const_true)
{
    EXPECT_TRUE((is_tuple<const std::tuple<int, char>&>::value));
}

TEST_F(type_traits_fixture, is_tuple_lref_true)
{
    EXPECT_TRUE((is_tuple<std::tuple<int, char>&>::value));
}

TEST_F(type_traits_fixture, is_tuple_const_lref_true)
{
    EXPECT_TRUE((is_tuple<const std::tuple<int, char>&>::value));
}

TEST_F(type_traits_fixture, is_tuple_false)
{
    EXPECT_FALSE(is_tuple<int>::value);
}

TEST_F(type_traits_fixture, is_tuple_rref_true)
{
    EXPECT_FALSE((is_tuple<std::tuple<int, char>&&>::value));
}

TEST_F(type_traits_fixture, is_tuple_const_rref_true)
{
    EXPECT_FALSE((is_tuple<const std::tuple<int, char>&&>::value));
}

} // namespace util
} // namespace ad
