#pragma once
#include "adfunction.hpp"
#include "utility.hpp"
#include <type_traits>

namespace ad {

	// Get Jacobian
	// Written Generically for any "Function"
	// Guarnatees compile-time substitution error on Jacobian_
	namespace core {

		// Jacobian_ Struct
		template <class F>
		struct Jacobian_ {};

		// Specialized Jacobian_ for Scalar Function 
		template <class ReturnType, class F>
		struct Jacobian_<Function<ReturnType, F>>
		{
			template <class Iter>
			static Iter compute(Iter it, Function<ReturnType, F> const& f)
			{
				std::for_each(f.x.begin(), f.x.end(),
					[&it](decltype(*(f.x.begin())) const& xi) mutable
				{*(it++) = *(xi.df_ptr); }
				);
				return it;
			}
		};

		// This is a generic jacobian_ core function
		// Parameter type F or Fs is generic (lambda or Function Obj)

		// Base case: One Scalar Function
		template <class F, class Iter>
		inline Iter jacobian_(Iter it, F&& f)
		{
			using F_pure = std::remove_cv_t<std::remove_reference_t<F>>;
			return Jacobian_<F_pure>::compute(it, std::forward<F>(f));
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
	template <class ReturnType, class F, class Iter
		, class = std::enable_if_t<utils::is_pointer_like_dereferenceable<Iter>::value, void>
	>
		inline void jacobian(Iter it, core::Function<ReturnType, F> const& f)
	{
		core::jacobian_(it, f);
	}

	// Vector Function Object
	template <class ReturnType, class... Fs, class Iter
		, class = std::enable_if_t<(sizeof...(Fs) > 1) && utils::is_pointer_like_dereferenceable<Iter>::value, void>
	>
		inline void jacobian(Iter it, core::Function<ReturnType, Fs...> const& f)
	{
		core::jacobian_(it, f.tup, std::make_index_sequence<sizeof...(Fs)>());
	}

	// ============================================================================
	// SPECIAL CASE: MATRIX
	// Matrix type must have the following implemented:
	// .zeros(m, n), .t(), .begin()

	template <class Matrix, class F>
	inline void jacobian_mat_helper(
		Matrix& mat
		, size_t m
		, size_t n
		, F&& f)
	{
		mat.zeros(m, n);
		jacobian(mat.begin(), std::forward<F>(f));
		mat = mat.t();
	}

	// Scalar Function Object
	template <class ReturnType, class F, class Matrix
		, class = typename std::enable_if<!core::is_Function<Matrix>::value, void>::type
	>
		inline auto jacobian(
			Matrix& mat
			, core::Function<ReturnType, F> const& f)
	{
		jacobian_mat_helper(mat, f.x.size(), 1, f);
	}

	// Vector Function Object
	template <class ReturnType, class Matrix, class... Fs
		, class = typename std::enable_if<(sizeof...(Fs) > 1) && !core::is_Function<Matrix>::value, void>::type
	>
		inline auto jacobian(
			Matrix& mat
			, core::Function<ReturnType, Fs...> const& f)
	{
		jacobian_mat_helper(mat, std::get<0>(f.tup).x.size(), std::tuple_size<decltype(f.tup)>::value, f);
	}

	// ============================================================================
	// Variadic on scalar lambda functions 
	// Same algorithm for both Matrix or Iter
	template <class ReturnType, class MatOrIter, class... Fs
		, class = typename std::enable_if<!core::is_Function<MatOrIter>::value, void>::type
	>
		inline auto jacobian(MatOrIter& mat_or_iter, Fs&&... fs)
	{
		auto f_obj = make_function<ReturnType>(std::forward<Fs>(fs)...);
		jacobian(mat_or_iter, f_obj);
	}

} // end ad 
