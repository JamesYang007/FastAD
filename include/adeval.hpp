#pragma once
#include <thread>
#include "adnode.hpp"

namespace ad {

	// Glue then evaluate
	// Forward propagation
	template <class ExprType>
	inline auto Evaluate(ExprType&& expr)
	{
		return expr.feval();
	}

	// Backward propagation
	template <class ExprType>
	inline void EvaluateAdj(ExprType&& expr)
	{
		expr.beval(1);
	}

	// Both forward and backward
	// This is primarily for details_tuple namespace usage
	// std::thread argument needs any overloaded function to be specified
	// workaround: create a non-overloaded fn and autodiff will call this
	template <class ExprType>
	inline auto EvaluateBoth(ExprType&& expr)
	{
		auto t = Evaluate(std::forward<ExprType>(expr));
		EvaluateAdj(std::forward<ExprType>(expr));
		return t;
	}

	template <class ExprType>
	inline auto autodiff(ExprType&& expr)
	{
		return EvaluateBoth(std::forward<ExprType>(expr));
	}

	//====================================================================================================

	// autodiff on tuple of expressions
	namespace details_tuple {

		// No multi-threading
		// Lvalue
		template <size_t I, class...ExprType>
		inline typename std::enable_if<I == sizeof...(ExprType), void>::type
			autodiff_(std::tuple<ExprType...>& tup) {}

		template <size_t I, class... ExprType>
		inline typename std::enable_if < I < sizeof...(ExprType), void>::type
			autodiff_(std::tuple<ExprType...>& tup)
		{
			ad::autodiff(std::get<I>(tup)); autodiff_<I + 1>(tup);
		}

		// multi-threading
		template <class ExprType>
		inline void autodiff_(ExprType& expr)
		{
			std::thread thr(ad::EvaluateBoth<ExprType>, expr);
			thr.join();
		}

		template <class ExprType1, class...ExprType>
		inline void autodiff_(ExprType1& expr1, ExprType&... expr)
		{
			std::thread thr(ad::EvaluateBoth<ExprType1>, expr1);
			autodiff_(expr...);
			thr.join();
		}

		// true_type
		template <class...ExprType, size_t...I>
		inline void autodiff_(std::tuple<ExprType...>& tup, std::index_sequence<I...>)
		{
			autodiff_(std::get<I>(tup)...);
		}

		template <class...ExprType>
		inline void autodiff_(std::tuple<ExprType...>& tup, std::true_type)
		{
			autodiff_(tup, std::make_index_sequence<sizeof...(ExprType)>());
		}

		// false_type
		template <class...ExprType>
		inline void autodiff_(std::tuple<ExprType...>& tup, std::false_type)
		{
			autodiff_<0>(tup);
		}

	}

	constexpr size_t THR_THRESHOLD = 1000;

	// USER-CALLABLE autodiff 
	template <class...ExprType>
	inline void autodiff(std::tuple<ExprType...>& tup)
	{
		details_tuple::autodiff_(tup
			, std::integral_constant<bool, (sizeof...(ExprType) >= THR_THRESHOLD)>());
	}

	template <class...ExprType>
	inline void autodiff(std::tuple<ExprType...>&& tup)
	{
		details_tuple::autodiff_(tup
			, std::integral_constant<bool, (sizeof...(ExprType) >= THR_THRESHOLD)>());
	}

} // end namespace ad
