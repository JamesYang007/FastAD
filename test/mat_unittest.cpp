#include <fastad_bits/mat.hpp>
#include "gtest/gtest.h"

namespace ad {

struct admat_fixture: ::testing::Test
{
protected:
    Mat<double> mat; 

    admat_fixture()
        : mat(4, 5, 0)
	{
		mat(0, 1) = 1;
		mat(1, 2) = 3;
		mat(2, 3) = 5;
		mat(3, 4) = 7;
	}
};

// copy constructor
TEST_F(admat_fixture, copy_constructor) {
	EXPECT_EQ(mat, Mat<double>(mat));
}

// iterator + size
TEST_F(admat_fixture, iter_size_comp) {
	size_t i = 0;
	Mat<double>::iterator it = mat.begin();
	while (it++ != mat.end() && ++i);
	EXPECT_EQ(i, mat.size());
}

// zeros (tests fill)
TEST_F(admat_fixture, zeros) {
	mat.zeros();
	Mat<double> z(mat.n_rows(), mat.n_cols(), 0);
	EXPECT_EQ(mat, z);
}

// transpose
TEST_F(admat_fixture, transpose) {
	Mat<double> t = mat.t();
	EXPECT_EQ(mat, t.t());
	EXPECT_EQ(t(1, 0), 1);
	EXPECT_EQ(t(2, 1), 3);
	EXPECT_EQ(t(3, 2), 5);
	EXPECT_EQ(t(4, 3), 7);
}

} // namespace 
