#include <fastad_bits/utility.hpp>
#include "gtest/gtest.h"

namespace utils {

struct utility_fixture: ::testing::Test 
{
protected:
};

TEST_F(utility_fixture, is_pointer_like_pointer)
{
    EXPECT_TRUE(is_pointer_like_dereferenceable<char*>::value);
}

TEST_F(utility_fixture, is_pointer_like_iterator)
{
    using iter_t = std::vector<int>::const_iterator;
    EXPECT_TRUE(is_pointer_like_dereferenceable<iter_t>::value);
}

TEST_F(utility_fixture, is_tuple_true)
{
    EXPECT_TRUE((is_tuple<std::tuple<int, char>>::value));
}

TEST_F(utility_fixture, is_tuple_const_true)
{
    EXPECT_TRUE((is_tuple<const std::tuple<int, char>&>::value));
}

TEST_F(utility_fixture, is_tuple_lref_true)
{
    EXPECT_TRUE((is_tuple<std::tuple<int, char>&>::value));
}

TEST_F(utility_fixture, is_tuple_const_lref_true)
{
    EXPECT_TRUE((is_tuple<const std::tuple<int, char>&>::value));
}

TEST_F(utility_fixture, is_tuple_false)
{
    EXPECT_FALSE(is_tuple<int>::value);
}

TEST_F(utility_fixture, is_tuple_rref_true)
{
    EXPECT_FALSE((is_tuple<std::tuple<int, char>&&>::value));
}

TEST_F(utility_fixture, is_tuple_const_rref_true)
{
    EXPECT_FALSE((is_tuple<const std::tuple<int, char>&&>::value));
}

} // namespace utils
