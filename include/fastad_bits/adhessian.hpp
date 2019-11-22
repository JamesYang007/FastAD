#pragma once
#include "adfunction.hpp"
#include "adforward.hpp"
#include "utility.hpp"
#include <type_traits>

// Hessian
// USER-CALLABLE ONLY WORKS WITH LAMBDA FUNCTION NOT FUNCTION OBJECT
namespace ad {
	namespace core {

		// Hessian struct
		template <class F>
		struct Hessian {};

		template <class ReturnType, class F>
		struct Hessian<Function<ForwardVar<ReturnType>, F>>
		{
			template <class Iter>
			static Iter compute_hess(Iter it, Function<ForwardVar<ReturnType>, F> const& f)
			{
				std::for_each(f.x.begin(), f.x.end(),
					[&it](decltype(*(f.x.begin())) const& xi) mutable
				{*(it++) = xi.df_ptr->df; }
				);
				return it;
			}

			template <class Iter>
			static void compute_grad(Iter it, Function<ForwardVar<ReturnType>, F> const& f)
			{
				std::for_each(f.x.begin(), f.x.end(),
					[&it](decltype(*(f.x.begin())) const& xi) mutable
				{*(it++) = xi.df_ptr->w; }
				);
			}
		};

		// Copies hessian column from hessian computation
		template <class F, class Iter>
		inline Iter hessian_(Iter it, F&& f)
		{
			using F_pure = std::remove_cv_t<std::remove_reference_t<F>>;
			return Hessian<F_pure>::compute_hess(it, std::forward<F>(f));
		}

		// Compute only the hessian
		template <class HessIter, class Iter, class F>
		inline auto hessian_(HessIter hess_begin, F&& f_hess, Iter begin, Iter end)
		{
			//using ReturnType = typename std::iterator_traits<Iter>::value_type;
			auto it = hess_begin;

			for (size_t i = 0; i < static_cast<size_t>(std::distance(begin, end)); ++i)
			{
				auto expr = f_hess(begin, end);
				f_hess.x[i].w.df = 1;
				autodiff(expr);
				// Record Hessian first column
				// Get iterator at next column of matrix
				it = core::hessian_(it, f_hess);
			}
		}

		// Copies gradient components from hessian computation
		template <class F, class Iter>
		inline void hessian_grad_(Iter it, F&& f)
		{
			using F_pure = std::remove_cv_t<std::remove_reference_t<F>>;
			Hessian<F_pure>::compute_grad(it, std::forward<F>(f));
		}

	} // end core

	// ===============================================================================================
	// Compute hessian from lambda function

	// Only Hessian
	template <class HessIter, class Iter, class F
		, class = std::enable_if_t<
		utils::is_pointer_like_dereferenceable<HessIter>::value &&
		utils::is_pointer_like_dereferenceable<Iter>::value
		>
	>
		inline void hessian(HessIter hess_begin, F&& f, Iter begin, Iter end)
	{
		using T = typename std::iterator_traits<Iter>::value_type;
		auto f_hess = make_function<ForwardVar<T>>(std::forward<F>(f));
		core::hessian_(hess_begin, f_hess, begin, end);
	}

	// Both Hessian and Gradient
	template <class HessIter, class GradIter, class Iter, class F
		, class = std::enable_if_t<
		utils::is_pointer_like_dereferenceable<HessIter>::value &&
		utils::is_pointer_like_dereferenceable<GradIter>::value &&
		utils::is_pointer_like_dereferenceable<Iter>::value
		>
	>
		inline void hessian(HessIter hess_begin, GradIter grad_begin, F&& f, Iter begin, Iter end)
	{
		using T = typename std::iterator_traits<Iter>::value_type;
		auto f_hess = make_function<ForwardVar<T>>(std::forward<F>(f));
		// Copy hessian
		core::hessian_(hess_begin, f_hess, begin, end);
		// Copy gradient component
		core::hessian_grad_(grad_begin, f_hess);
	}


	// ============================================================================
	// SPECIAL CASE
	// Matrix type must have the following implemented:
	// .zeros(m, n), .begin()
	template <class Matrix, class F, class Iter
		, class = std::enable_if_t<
		utils::is_pointer_like_dereferenceable<Iter>::value
		>
	>
		inline void hessian(Matrix& hess_mat, F&& f, Iter begin, Iter end)
	{
		const size_t n = static_cast<size_t>(std::distance(begin, end));
		hess_mat.zeros(n, n);
		hessian(hess_mat.begin(), std::forward<F>(f), begin, end);
	}

	template <class Matrix, class F, class Iter
		, class = std::enable_if_t<
		utils::is_pointer_like_dereferenceable<Iter>::value
		>
	>
		inline void hessian(Matrix& hess_mat, Matrix& grad_mat, F&& f, Iter begin, Iter end)
	{
		const size_t n = static_cast<size_t>(std::distance(begin, end));
		hess_mat.zeros(n, n);
		grad_mat.zeros(1, n);
		hessian(hess_mat.begin(), grad_mat.begin(), std::forward<F>(f), begin, end);
	}

} // end ad
