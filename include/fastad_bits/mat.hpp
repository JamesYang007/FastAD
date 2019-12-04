#pragma once
#include <iostream>
#include <vector>
#include <algorithm>

namespace ad {

template <class T>
class Mat;
template <class T>
std::ostream& operator<<(std::ostream& os, const Mat<T>& mat);
template <class T>
bool operator==(const Mat<T>& mat1, const Mat<T>& mat2);

template <class T>
class Mat {

public:

	// constructors
	Mat() = default;
	Mat(const Mat& mat) 
		: data_(mat.data_), rows_(mat.rows_), cols_(mat.cols_) 
	{}
	Mat(size_t size, const T& fill) 
		: data_(size*size, fill), rows_(size), cols_(size) 
	{}
	Mat(size_t rows, size_t cols, const T& fill) 
		: data_(rows*cols, fill), rows_(rows), cols_(cols) 
	{}

	// operators
	T& operator()(size_t row, size_t col) 
	{ 
		return data_[row*cols_ + col];  
	}
	const T& operator()(size_t row, size_t col) const 
	{ 
		return const_cast<Mat&>(*this)(row, col); 
	}
	friend std::ostream& operator<< <>(std::ostream& os, const Mat& mat);
	friend bool operator== <>(const Mat& mat1, const Mat& mat2);
	friend bool operator!=(const Mat& mat1, const Mat& mat2) 
	{ 
		return !(mat1 == mat2); 
	}

	// return size details from private members
	size_t size() const 
	{ 
		return rows_*cols_; 
	}
	size_t n_rows() const 
	{ 
		return rows_; 
	}
	size_t n_cols() const 
	{ 
		return cols_; 
	}

	// print to stdout with given header
	void print(const std::string& header) 
	{ 
		std::cout << header << std::endl << *this; 
	}

	// fill the matrix (will resize for new dimensions)
	void fill(size_t rows, size_t cols, const T& fill);
	void fill(const T& fill) {
		fill(this->rows_, this->cols_, fill);
	}
	void zeros(size_t rows, size_t cols) 
	{ 
		fill(rows, cols, 0); 
	}	

	// transpose
	Mat t();	

	// iterator just uses std::vector's
	using iterator = typename std::vector<T>::iterator;
	using const_iterator = typename std::vector<T>::const_iterator;

	iterator begin() 
	{ 
		return data_.begin(); 
	}
	const_iterator begin() const 
	{ 
		return data_.begin(); 
	}
	iterator end() 
	{ 
		return data_.end(); 
	}
	const_iterator end() const 
	{ 
		return data_.end(); 
	}
	
private:

	std::vector<T> data_;	
	size_t rows_, cols_;
};

// matrix -> string format is tab-separated by column, newline separated by row
template <class T>
std::ostream& operator<<(std::ostream& os, const Mat<T>& mat) 
{
	size_t i = 0;
	for (const T& item : mat) {
		os << item << '\t';
		if (++i >= mat.cols_) { 
			os << std::endl; i = 0; 
		}
	}
	return os;
}

// matrices are equal if dimensions are the same and entries are identical
template <class T>
bool operator==(const Mat<T>& mat1, const Mat<T>& mat2) 
{
	if (mat1.rows_ != mat2.rows_ || mat1.cols_ != mat2.cols_) { 
		return false; 
	}	

	typename Mat<T>::const_iterator it = mat2.begin();
	for (const T& item : mat1) {
		if (item != *it++) {
			return false; 
		}
	}

	return true;
}

template <class T>
void Mat<T>::fill(size_t rows, size_t cols, const T& fill) {
	if (this->size() != rows*cols) {
		this->data_.resize(rows*cols);
	}
	std::fill(this->begin(), this->end(), fill);
}

template <class T>
Mat<T> Mat<T>::t() 
{
	Mat<T> mat_t(this->cols_, this->rows_, 0);

	for (size_t r = 0; r < this->rows_; r++) {
		for (size_t c = 0; c < this->cols_; c++) {
			mat_t(c, r) = (*this)(r, c);
		}
	}

	return mat_t;
}

}
