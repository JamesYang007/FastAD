#pragma once
#include <ostream>
#include <vector>

namespace ad {

template <typename T>
class Matrix;
template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat);

template <typename T>
class Matrix {

public:

	Matrix() : data() {}
	Matrix(const Matrix& mat) : data(mat.data) {}
	Matrix(const std::vector< std::vector<T> >& dat) : data(dat) {}
	Matrix(size_t size, T fill) : data(size, std::vector<T>(size, fill)) {}
	Matrix(size_t rows, size_t cols, T fill) : data(rows, std::vector<T>(cols, fill)) {}
	~Matrix() {}

	friend std::ostream& operator<< <T>(std::ostream& os, const Matrix<T>& mat);
	std::vector<T>& operator[](int i) { return data[i];  }
	const std::vector<T>& operator[](int i) const { return ((Matrix<T>&)*this)[i]; }

	inline void fill(size_t rows, size_t cols, T fill) { *this = Matrix<T>(rows, cols, fill); }
	inline void zeros(size_t rows, size_t cols) { this->fill(rows, cols, 0); }

	class iterator {

	public:

		iterator(typename std::vector<T>::iterator it, typename std::vector< std::vector<T> >::iterator row_it) { iter = it; row_iter = row_it; }
		~iterator() {}
		iterator& operator=(const iterator& it) { iter = it.iter; row_iter = it.row_iter; }
		iterator& operator++() { if (std::distance(++iter, row_iter->end()) <= 0) { iter = (++row_iter)->begin(); } return *this; }
		iterator operator++(int) { iterator cur = *this; ++(*this); return cur; }
		T& operator*() const { return *iter; }
		bool operator==(const iterator& it) const { return row_iter == it.row_iter; }
		bool operator!=(const iterator& it) const { return !(*this == it); }

	private:

		typename std::vector<T>::iterator iter;				
		typename std::vector< std::vector<T> >::iterator row_iter;

	};

	iterator begin() { return iterator(data.begin()->begin(), data.begin()); }
	iterator end() { return iterator((data.end() - 1)->end(), data.end()); }
	const iterator begin() const { return ((Matrix<T>&)*this).begin(); }
	const iterator end() const { return ((Matrix<T>&)*this).end(); }

private:

	std::vector< std::vector<T> > data;	

};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
	for (auto& row : mat.data) {
		for (auto& item : row) { os << item << '\t'; }
		os << std::endl;
	}
	return os;
}

}
