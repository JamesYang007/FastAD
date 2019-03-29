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

		// Function Base
		struct FunctionBase
		{
			size_t n_func;
			FunctionBase(size_t n_func_)
				: n_func(n_func_)
			{}
		};

		// Scalar Function Base
		template <class ReturnType>
		struct ScalarFunctionBase : FunctionBase
		{
			// Store values and gradient components
			Vec<ReturnType> x;
			// Needed to hold evaluation of each expression within tuple of expressions
			Vec<ReturnType> w;

			ScalarFunctionBase()
				: x(0), w(0), FunctionBase(1)
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

		// =====================================================================================
		// =====================================================================================
		// =====================================================================================
		// MODIFY VECTOR FIELD
		// Vector Function Base

		struct VectorFunctionBase : FunctionBase
		{
			VectorFunctionBase(size_t n)
				: FunctionBase(n)
			{}

			// Zip returns of each f_i(begin, end)
			template <class Tup, class Iter, size_t... I>
			static inline auto zip_func(Tup&& tup, Iter begin, Iter end, std::index_sequence<I...>) {
				return std::make_tuple(std::get<I>(tup)(begin, end)...);
			}

			template <class Iter, class...Fs>
			static inline auto zip_func(std::tuple<Fs...>& tup, Iter begin, Iter end) {
				return zip_func(tup, begin, end
					, std::make_index_sequence<sizeof...(Fs)>()
				);
			}

			// Zip functions using expr_eval(Vec<T>&)
			template <class Tup, class T, size_t...I>
			static inline auto zip_func(Tup&& tup, Vec<T>& x, std::index_sequence<I...>)
			{
				return std::make_tuple(std::get<I>(tup).expr_eval(x)...);
			}

			template <class T, class...Fs>
			static inline auto zip_func(std::tuple<Fs...>& tup, Vec<T>& x)
			{
				return zip_func(tup, x, std::make_index_sequence<sizeof...(Fs)>());
			}
		};

		// Vector Field
		// Wrapper of tuple of scalar functions
		// If there are more than threshold number of scalar functions, we use multi-threading
		template <class ReturnType, class ...Fs>
		struct Function : VectorFunctionBase
		{
			using base_type = VectorFunctionBase;
			std::tuple<Function<ReturnType, Fs>...> tup;
			Function(Fs&&... fs)
				: tup(std::make_tuple(Function<ReturnType, Fs>(std::forward<Fs>(fs))...))
				, base_type(sizeof...(Fs))
			{}

			Function(Function<ReturnType, Fs> const&... fs)
				: tup(std::make_tuple(fs...))
				, base_type(sizeof...(Fs))
			{}

			// Returns tuple of return expression from each scalar function
			template <class Iter>
			inline auto operator()(Iter begin, Iter end) {
				return base_type::zip_func(this->tup, begin, end);
			}

			template <class T>
			inline auto expr_eval(Vec<T>& x)
			{
				return base_type::zip_func(this->tup, x);
			}

		};

		// =====================================================================================
		// =====================================================================================
		// =====================================================================================

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

			// glue with vector w, and tuple of expressions
			template <class T, class... ExprType>
			inline auto glue_many(ad::Vec<T>& w, std::tuple<ExprType...>& tup)
			{
				return glue_many(w, tup, std::make_index_sequence<sizeof...(ExprType)>());
			}

			// glue with vector w, and one expression
			template <class T, class ExprType>
			inline auto glue_many(ad::Vec<T>& w, ExprType&& expr)
			{
				auto tup = std::make_tuple(std::forward<ExprType>(expr));
				return glue_many(w, tup);
			}

		} // end details

		template <class ReturnType, class F>
		struct Function<ReturnType, F> : ScalarFunctionBase<ReturnType>
		{
			// Lambda Function
			F f;

			Function(F&& f)
				: ScalarFunctionBase<ReturnType>(), f(f)
			{}

			// Returns GlueNode glueing w[i] = expr_i
			template <class Iter>
			inline auto operator()(Iter begin, Iter end) {
				this->init_x(this->x, begin, end);
				return expr_eval(this->x);
			}

			// Returns correct expression reprsenting f(x)
			template <class T>
			inline auto expr_eval(Vec<T>& x)
			{
				// Initialize this->w
				constexpr size_t w_capacity = std::tuple_size<
					std::result_of_t<F(Vec<ReturnType>&, Vec<ReturnType>&)>
				>::value;
				this->init_w(this->w, w_capacity);
				// Create expression
				auto&& exprs = this->f(x, this->w);
				return details::glue_many(this->w, exprs);
			}

		};

		// Composition: Vector Function of Vector Function
		template <class ReturnType, class FComposer, class FComposed
			, bool scalar = std::is_base_of_v<
			ScalarFunctionBase<ReturnType>
			, std::remove_reference_t<FComposer> // VERY IMPORTANT TO REMOVE REF
			>
		>
			struct ComposedFunction : VectorFunctionBase
		{

		};


		// Composition: Scalar Function of Vector Function
		template <class ReturnType, class FComposer, class FComposed>
		struct ComposedFunction <ReturnType, FComposer, FComposed, true>
			: ScalarFunctionBase<ReturnType>
		{
			using base_type = ScalarFunctionBase<ReturnType>;
			FComposed composed;
			FComposer composer;

			// Copy construct is OK because before any calculations, this->x/this->w will be cleared.
			ComposedFunction(FComposer&& fcomposer, FComposed&& fcomposed)
				: composer(std::forward<FComposer>(fcomposer))
				, composed(std::forward<FComposed>(fcomposed))
			{}

			// Return expression representing FComposer(FComposed)
			template <class Iter
				, class T = typename std::iterator_traits<Iter>::value_type
			>
				inline auto operator()(Iter begin, Iter end) {
				base_type::init_x(this->x, begin, end);
				return expr_eval(this->x);
			}

			template <class T>
			inline auto expr_eval(Vec<T>& x)
			{
				// Initialize this->w
				base_type::init_w(this->w, this->composed.n_func);
				// Get correct expr from composed function
				// This may be 1 expression or tuple of expressions
				// Output type does not matter since glue_many has respective overloads
				auto&& expr_composed = this->composed.expr_eval(x);
				// Glue nodes with this->w
				auto&& expr_composed_glued = details::glue_many(this->w, expr_composed);
				// Get correct expr from composer function
				auto&& expr_composer = this->composer.expr_eval(this->w);
				// Finally, glue expr_composed_glued, expr_composer
				// Note that expr_composer must be a single expression by specialization
				return details::glue_many(expr_composed_glued, expr_composer);
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
	template <class ReturnType = double, class... Fs>
	inline auto make_function(Fs&&... fs)
	{
		return core::Function<ReturnType, Fs...>(std::forward<Fs>(fs)...);
	}

	// =====================================================================================
	// Make ComposedFunction with Function Objects

	// F o G
	template <class ReturnType = double, class FComposer, class FComposed>
	inline auto compose(FComposer&& f, FComposed&& g)
	{
		return core::ComposedFunction<ReturnType, FComposer, FComposed>
			(std::forward<FComposer>(f), std::forward<FComposed>(g));
	}

	// F1 o F2 o ... o Fn
	template <class ReturnType = double, class F, class... Fs>
	inline auto compose(F&& f, Fs&&... fs)
	{
		return compose(std::forward<F>(f)
			, compose(std::forward<Fs>(fs)...));
	}

} // end ad
