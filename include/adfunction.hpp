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

		// Scalar Function Base
		template <class ReturnType>
		struct ScalarFunctionBase
		{
			// Store values and gradient components
			Vec<ReturnType> x;
			// Needed to hold evaluation of each expression within tuple of expressions
			Vec<ReturnType> w;

			ScalarFunctionBase()
				: x(0), w(0)
			{}

			// Initializes x
			template <class Iter, class V
				, class T = typename std::iterator_traits<Iter>::value_type
			>
				static inline auto init_x(V& x, Iter begin, Iter end) {
				x.clear();
				std::for_each(begin, end, [&x](T const& x_i) mutable
				{x.emplace_back(x_i); }
				);
			}

			// Initialize w
			template <class V>
			static inline auto init_w(V& w, size_t w_capacity) {
				w.clear();
				w.resize(w_capacity);
			}

		};

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

		// Details to compute_expr
		// Call lambda function to create tuple of RHS expressions
		// Glue w[i] = RHS_i
		namespace details {

			// Glue many nodes (equiv: (expr1, expr2, ..., exprn))
			template <class ExprType>
			inline auto glue_many(ExprType&& expr)
			{
				return std::forward<ExprType>(expr);
			}

			template <class ExprType1, class... ExprType
				, class = std::enable_if_t<(sizeof...(ExprType) > 0)>
			>
				inline auto glue_many(ExprType1&& expr1, ExprType&&... exprs)
			{
				return (expr1, glue_many(std::forward<ExprType>(exprs)...));
			}

			// w[0] = expr_0, w[1] = expr_1, ...
			template <class T, class... ExprType, size_t... I>
			inline auto glue_many(ad::Vec<T>& w, std::tuple<ExprType...>& tup
				, std::index_sequence<I...>)
			{
				return glue_many((w[I] = std::get<I>(tup))...);
			}

			template <class T, class... ExprType>
			inline auto glue_many(ad::Vec<T>& w, std::tuple<ExprType...>& tup)
			{
				return glue_many(w, tup, std::make_index_sequence<sizeof...(ExprType)>());
			}

			// Compute full expression from lambda function
			template <class T, class F>
			inline auto compute_expr(ad::Vec<T>& x, ad::Vec<T>& w, F&& f)
			{
				auto&& exprs = f(x, w);
				return glue_many(w, exprs);
			}

		} // end details

		template <class ReturnType, class F>
		struct Function<ReturnType, F> : ScalarFunctionBase<ReturnType>
		{
			// Lambda Function
			F f;

			Function(F const& f)
				: ScalarFunctionBase<ReturnType>(), f(f)
			{}

			// Returns GlueNode glueing w[i] = expr_i
			template <class Iter>
			inline auto operator()(Iter begin, Iter end) {
				constexpr size_t w_capacity = std::tuple_size<
					typename std::result_of<
					F(Vec<ReturnType>&, Vec<ReturnType>&)
					>::type
				>::value;
				this->init_x(this->x, begin, end);
				this->init_w(this->w, w_capacity);
				return details::compute_expr(this->x, this->w, this->f);
			}

		};


		// Composition: Vector Function of Vector Function
		template <class ReturnType, class FComposed, class FComposer
			, bool scalar = std::is_base_of_v<ScalarFunctionBase<ReturnType>, FComposer>
		>
			struct ComposedFunction;


		// Composition: Scalar Function of Vector Function
		template <class ReturnType, class FComposed, class FComposer>
		struct ComposedFunction <ReturnType, FComposed, FComposer, true>
			: ScalarFunctionBase<ReturnType>
		{
			FComposed composed;
			FComposer composer;

			// Copy construct is OK because before any calculations, this->x/this->w will be cleared.
			ComposedFunction(FComposer const& fcomposer, FComposed const& fcomposed)
				: composer(fcomposer), composed(fcomposed)
			{}

			template <class Iter
				, class T = typename std::iterator_traits<Iter>::value_type
			>
				inline auto operator()(Iter begin, Iter end) {
				this->init_x(this->x, begin, end);
				this->init_w(this->w, 1);
				this->compute_expr(this->x, composed.w, composed.f);
				return details::compute_expr(this->x, this->w, this->f);
			}

			template <class T>
			inline auto operator()(Vec<T>& x, Vec<T>& w)
			{
				auto&& expr = composed(x, w);
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
