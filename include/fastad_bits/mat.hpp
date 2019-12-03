#pragma once
#include <iostream>
#include <vector>

namespace ad {

template <typename T>
class Mat;
template <typename T>
std::ostream& operator<<(std::ostream& os, const Mat<T>& mat);

template <typename T>
class Mat {

public:

	Mat() : data() {}
	Mat(const Mat& mat) : data(mat.data) {}
	Mat(const std::vector< std::vector<T> >& dat) : data(dat) {}
	Mat(size_t size, T fill) : data(size, std::vector<T>(size, fill)) {}
	Mat(size_t rows, size_t cols, T fill) : data(rows, std::vector<T>(cols, fill)) {}
	~Mat() {}

	friend std::ostream& operator<< <T>(std::ostream& os, const Mat<T>& mat);
	std::vector<T>& operator[](int i) { return data[i];  }
	const std::vector<T>& operator[](int i) const { return ((Mat<T>&)*this)[i]; }

	const inline std::vector< std::vector<T> >& vecs() const { return data; }
	inline size_t rows() const { return data.size(); }
	inline size_t cols() const { return data.begin() != data.end() ? data.begin()->size() : 0; }	

	inline void print(std::ostream& os, std::string header) { os << header << std::endl << *this; }
	inline void print(std::string header) { print(std::cout, header);  }

	inline void fill(size_t rows, size_t cols, T fill) { *this = Mat<T>(rows, cols, fill); }
	inline void zeros(size_t rows, size_t cols) { fill(rows, cols, 0); }	

	Mat t();	

	class iterator {

	public:

		iterator(typename std::vector<T>::iterator it, typename std::vector< std::vector<T> >::iterator row_it, typename std::vector< std::vector<T> >::iterator row_it_end)
			: iter(it), row_iter(row_it), row_end_iter(row_it_end) {}
		~iterator() {}
		iterator& operator=(const iterator& it) { iter = it.iter; row_iter = it.row_iter; row_end_iter = it.row_end_iter; return *this; }
		iterator& operator++() { if (++iter == row_iter->end() && ++row_iter != row_end_iter) { iter = row_iter->begin(); } return *this; }
		T& operator*() const { return *iter; }
		bool operator==(const iterator& it) const { return row_iter == it.row_iter && iter == it.iter; }
		bool operator!=(const iterator& it) const { return !(*this == it); }

	private:

		typename std::vector<T>::iterator iter;				
		typename std::vector< std::vector<T> >::iterator row_iter;
		typename std::vector< std::vector<T> >::iterator row_end_iter;

	};

	iterator begin() { return iterator(data.begin()->begin(), data.begin(), data.end()); }
	iterator end() { return iterator((data.end() - 1)->end(), data.end(), data.end()); }
	const iterator begin() const { return ((Mat<T>&)*this).begin(); }
	const iterator end() const { return ((Mat<T>&)*this).end(); }

private:

	std::vector< std::vector<T> > data;	

};

template <typename T>
Mat<T> Mat<T>::t() {
	std::vector< std::vector<T> > cols(this->cols(), std::vector<T>());

	typename std::vector< std::vector<T> >::iterator col_it;
	for (const std::vector<T>& row : this->vecs()) {
		col_it = cols.begin();
		for (const T& item : row) {
			col_it->push_back(item);
			++col_it;
		}
	}

	return Mat<T>(cols);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Mat<T>& mat) {
	for (const std::vector<T>& row : mat.vecs()) {
		for (const T& item : row) { os << item << '\t'; }
		os << std::endl;
	}
	return os;
}

}
