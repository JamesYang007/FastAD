#pragma once
#include "advec.hpp"

// USER-FRIENDLY MAKE_LMDA
#define MAKE_LMDA(...) \
[](auto& x, auto& w) \
{\
return std::make_tuple(##__VA_ARGS__);\
}


// Function Objects
namespace ad {

	namespace core {

		// Vector Field
		// Wrapper of tuple of scalar functions
		// If there are more than threshold number of scalar functions, we use multi-threading
		template <class ReturnType, class ...Fs>
		struct Function
		{
			std::tuple<Function<ReturnType, Fs>...> tup;
			Function(Fs const&... fs)
				: tup(std::make_tuple(Function<ReturnType, Fs>(std::forward<Fs>(fs))...))
			{}

			Function(Function<ReturnType, Fs> const&... fs)
				: tup(std::make_tuple(fs...))
			{}

			// Returns tuple of return expression from each scalar function
			template <class Iter>
			inline auto operator()(Iter begin, Iter end) {
				return zip_func(begin, end);
			}

		private:

			// Zip returns of each f_i(begin, end)
			template <class Iter, size_t... I>
			auto zip_func(Iter begin, Iter end, std::index_sequence<I...>) {
				return std::make_tuple(std::get<I>(this->tup)(begin, end)...);
			}

			template <class Iter>
			auto zip_func(Iter begin, Iter end) {
				return zip_func(begin, end
					, std::make_index_sequence<sizeof...(Fs)>()
				);
			}
		};

		// Scalar Function (specialization of Vector Field)
		template <class ReturnType, class F>
		struct Function<ReturnType, F>
		{
			Vec<ReturnType> x;
			Vec<ReturnType> w;
			F f;

			Function(F const& f)
				: f(f), x(0), w(0)
			{}

			// Returns GlueNode glueing w[i] = expr_i
			template <class Iter
				, class T = typename std::iterator_traits<Iter>::value_type
			>
				inline auto operator()(Iter begin, Iter end) {
				this->x.clear();
				std::for_each(begin, end, [this](T const& x_i) mutable
				{this->x.emplace_back(x_i); }
				);
				constexpr size_t w_capacity = std::tuple_size<
					typename std::result_of<
					F(Vec<ReturnType>&, Vec<ReturnType>&)
					>::type
				>::value;
				this->w.clear();
				this->w.resize(w_capacity);
				auto&& exprs = f(this->x, this->w);
				return glue_many(exprs);
			}

			// Assuming x, w have already been initialized
			inline auto operator()() {
				auto&& exprs = f(this->x, this->w);
				return glue_many(exprs);
			}


		private:

			// Glue many nodes (equiv: (expr1, expr2, ..., exprn))
			template <class ExprType>
			inline auto glue_many(ExprType const& expr)
			{
				return expr;
			}

			template <class ExprType1, class... ExprType
				, class = typename std::enable_if<(sizeof...(ExprType) > 0), void>::type
			>
				inline auto glue_many(ExprType1&& expr1
					, ExprType&&... exprs)
			{
				return (expr1, glue_many(std::forward<ExprType>(exprs)...));
			}

			// w[0] = expr_0, w[1] = expr_1, ...
			template <class... ExprType, size_t... I>
			inline auto glue_many(std::tuple<ExprType...> const& tup
				, std::index_sequence<I...>)
			{
				return glue_many((this->w[I] = std::get<I>(tup))...);
			}

			template <class... ExprType>
			inline auto glue_many(std::tuple<ExprType...> const& tup)
			{
				return glue_many(tup, std::make_index_sequence<sizeof...(ExprType)>());
			}

		};

		// Helper is_Function
		template <class T>
		struct is_Function
		{
			static constexpr bool value = false;
		};

		template <class ReturnType, class F>
		struct is_Function<Function<ReturnType, F>>
		{
			static constexpr bool value = true;
		};


	} // namespace core

	// Make Function Object with lambda functions
	template <class ReturnType, class... Fs>
	inline auto make_function(Fs&&... fs)
	{
		return core::Function<ReturnType, Fs...>(std::forward<Fs>(fs)...);
	}


} // end ad
