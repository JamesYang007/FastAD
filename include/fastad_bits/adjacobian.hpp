#pragma once
#include "adfunction.hpp"
#include "adeval.hpp"
#include "utility.hpp"
#include <type_traits>

namespace ad {

	// Get Jacobian
	// Written Generically for any "Function"
	namespace core {

		template <class ReturnType, class Iter>
		inline Iter compute(Iter it, ScalarFunctionBase<ReturnType> const& f)
		{
			std::for_each(f.x.begin(), f.x.end(),
				[&it](decltype(*(f.x.begin())) const& xi) mutable
			{*(it++) = *(xi.df_ptr); }
			);
			return it;
		}

		// This is a generic jacobian_ core function
		// Parameter type F or Fs is generic (lambda or Function Obj)

		// Base case: One Scalar Function
		template <class F, class Iter>
		inline Iter jacobian_(Iter it, F&& f)
		{
			return compute(it, std::forward<F>(f));
		}

		// Variadic on Scalar Function
		template <class F, class Iter, class... Fs
			, class = std::enable_if_t<(sizeof...(Fs) > 0)>
				>
			inline void jacobian_(Iter it, F&& f, Fs&&... fs)
		{
			it = jacobian_(it, std::forward<F>(f));
			jacobian_(it, std::forward<Fs>(fs)...);
		}

		// Tuple of Scalar Function
		template <class Iter, class... Fs, size_t... I>
		inline void jacobian_(
			Iter it
			, std::tuple<Fs...> const& tup
			, std::index_sequence<I...>)
		{
			jacobian_(it, std::get<I>(tup)...);
		}

	} // end core

	// ============================================================================
	/* Using generic iterator */
	// Iterator must iterate over elements column by column

	// Scalar Function Object
	template <class ReturnType, class Iter
		, class = std::enable_if_t<utils::is_pointer_like_dereferenceable<Iter>::value>
	>
		inline void jacobian(Iter it, core::ScalarFunctionBase<ReturnType> const& f)
	{
		core::jacobian_(it, f);
	}

	// Vector Function Object
	template <class ReturnType, class... Fs, class Iter
		, class = std::enable_if_t<utils::is_pointer_like_dereferenceable<Iter>::value>
	>
		inline void jacobian(Iter it, core::VectorFunctionBase<ReturnType, Fs...> const& f)
	{
		core::jacobian_(it, f.tup, std::make_index_sequence<sizeof...(Fs)>());
	}

	// ============================================================================
	// SPECIAL CASE: MATRIX
	// Matrix type must have the following implemented:
	// .zeros(m, n), .t(), .begin()

	template <class Matrix, class F>
	inline void jacobian_mat_helper(Matrix& mat, size_t m, size_t n, F&& f)
	{
		mat.zeros(m, n);
		jacobian(mat.begin(), std::forward<F>(f));
		mat = mat.t();
	}

	// Scalar Function Object
	template <class ReturnType, class Matrix
		, class = std::enable_if_t<!core::is_Function<Matrix>::value>
	>
		inline auto jacobian(
			Matrix& mat
			, core::ScalarFunctionBase<ReturnType> const& f)
	{
		jacobian_mat_helper(mat, f.x.size(), 1, f);
	}

	// Vector Function Object
	template <class ReturnType, class Matrix, class... Fs
		, class = std::enable_if_t<!core::is_Function<Matrix>::value>
	>
		inline auto jacobian(
			Matrix& mat
			, core::VectorFunctionBase<ReturnType, Fs...> const& f)
	{
		jacobian_mat_helper(mat, std::get<0>(f.tup).x.size(), f.n_func, f);
	}

	// ============================================================================
	// Variadic on scalar lambda functions 
	// Same algorithm for both Matrix or Iter
	template <class ReturnType = double, class MatOrIter, class Iter, class... Fs
		, class = std::enable_if_t<
		!core::is_Function<MatOrIter>::value &
		utils::is_pointer_like_dereferenceable<Iter>::value
		>
	>
		inline auto jacobian(MatOrIter& mat_or_iter, Iter begin, Iter end, Fs&&... fs)
	{
		auto f_obj = make_function<ReturnType>(std::forward<Fs>(fs)...);
		autodiff(f_obj(begin, end));
		jacobian(mat_or_iter, f_obj);
	}

} // end ad 