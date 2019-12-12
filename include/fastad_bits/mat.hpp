#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#define DEFAULT_PRINT_FIELD_WIDTH 8

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
	Mat()
		: data_(), rows_(0), cols_(0)
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

	// print (formatted with proper spacing) to stdout with given header
	void print_at_width(const std::string& header, unsigned int field_width) const;
	void print(const std::string& header) const
	{
		print_at_width(header, DEFAULT_PRINT_FIELD_WIDTH);
	}
	std::enable_if_t<std::is_floating_point_v<T>> print_at_precision(const std::string& header, unsigned int precision) const;

	// fill the matrix (will resize for new dimensions)
	void fill(size_t rows, size_t cols, const T& fill);

	void fill(const T& fill) 
    {
		this->fill(this->rows_, this->cols_, fill);
	}

	void zeros(size_t rows, size_t cols) 
	{ 
		this->fill(rows, cols, 0); 
	}	
	void zeros()
	{
		this->zeros(this->rows_, this->cols_);
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
			os << std::endl; 
            i = 0; 
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
void Mat<T>::print_at_width(const std::string& header, unsigned int field_width) const
{ 
	std::cout << header << std::endl;
	std::cout << std::setfill(' ');	
	size_t i = 0;
	for (const T& item : *this) {
		std::cout << std::setw(field_width) << item;
		if (++i >= this->cols_) { 
			std::cout << std::endl; i = 0; 
		}
	}
}

template <class T>
std::enable_if_t<std::is_floating_point_v<T>> Mat<T>::print_at_precision(const std::string& header, unsigned int precision) const {
	std::cout.precision(precision);
	std::cout << std::defaultfloat;
	print_at_width(header, precision+8);
}

template <class T>
void Mat<T>::fill(size_t rows, size_t cols, const T& fill) 
{
	if (this->size() != rows*cols) {
		this->data_.resize(rows*cols);
	}
	this->rows_ = rows;
	this->cols_ = cols;
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
