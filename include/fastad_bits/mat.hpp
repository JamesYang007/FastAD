#pragma once
#include <iostream>
#include <vector>

namespace ad {

template <typename T>
class Mat;
template <typename T>
std::ostream& operator<<(std::ostream& os, const Mat<T>& mat);
template <typename T>
bool operator==(const Mat<T>& mat1, const Mat<T>& mat2);

template <typename T>
class Mat {

public:

	Mat() : data(), rows_(0), cols_(0) {}
	Mat(const Mat& mat) : data(mat.data), rows_(mat.rows_), cols_(mat.cols_) {}
	Mat(size_t size, T fill) : data(size*size, fill), rows_(size), cols_(size) {}
	Mat(size_t rows, size_t cols, T fill) : data(rows*cols, fill), rows_(rows), cols_(cols) {}

	friend std::ostream& operator<<<>(std::ostream& os, const Mat& mat);
	friend bool operator==<>(const Mat& mat1, const Mat& mat2);
	friend bool operator!=(const Mat& mat1, const Mat& mat2) { return !(mat1 == mat2); }
	T& operator()(size_t row, size_t col) { return data[row*cols_ + col];  }
	const T& operator()(size_t row, size_t col) const { return ((Mat&)*this)(row, col); }

	size_t size() const { return rows_*cols_; }
	size_t n_rows() const { return rows_; }
	size_t n_cols() const { return cols_; }

	void print(std::ostream& os, std::string header) { os << header << std::endl << *this; }
	void print(std::string header) { print(std::cout, header);  }

	void fill(size_t rows, size_t cols, T fill) { *this = Mat(rows, cols, fill); }
	void zeros(size_t rows, size_t cols) { fill(rows, cols, 0); }	

	Mat t();	

	using iterator = typename std::vector<T>::iterator;
	using const_iterator = typename std::vector<T>::const_iterator;

	iterator begin() { return data.begin(); }
	const_iterator begin() const { return data.begin(); }
	iterator end() { return data.end(); }
	const_iterator end() const { return data.end(); }
	
private:

	Mat(const std::vector<T>& dat, size_t rows, size_t cols) : data(dat), rows_(rows), cols_(cols) {}

	std::vector<T> data;	
	size_t rows_, cols_;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Mat<T>& mat) {
	size_t i = 0;
	for (const T& item : mat) {
		os << item << '\t';
		if (++i >= mat.cols_) { os << std::endl; i = 0; }
	}
	return os;
}

template <typename T>
bool operator==(const Mat<T>& mat1, const Mat<T>& mat2) {
	if (mat1.rows_ != mat2.rows_ || mat1.cols_ != mat2.cols_) { return false; }	

	typename Mat<T>::const_iterator it = mat2.begin();
	for (const T& item : mat1) {
		if (item != *it++) { return false; }
	}

	return true;
}

template <typename T>
Mat<T> Mat<T>::t() {
	std::vector<T> data_t;

	Mat<T>::iterator it;
	for (size_t c = 0; c < this->cols_; c++) {
		it = this->begin() + c;
		while (std::distance(it, this->end()) > 0) {
			data_t.push_back(*it);
			it += this->cols_;
		}
	}

	return Mat<T>(data_t, this->cols_, this->rows_);
}

}
