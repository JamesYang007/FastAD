#pragma once
#include <tuple>
#include "advec.hpp"
#include "utility.hpp"

// USER-FRIENDLY MAKE_LMDA
#define MAKE_LMDA(...) \
[](auto& x, auto& w) \
{\
return std::make_tuple(__VA_ARGS__);\
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
				: FunctionBase(1), x(0), w(0)
			{}

			// Initialize x
			template <class Iter
				, class T = typename std::iterator_traits<Iter>::value_type
			>
				static inline auto init_x(Vec<T>& x, Iter begin, Iter end) {
				x.clear();
				std::for_each(begin, end, [&x](T const& x_i) mutable
				{x.emplace_back(x_i); }
				);
			}

			// Initialize w
			template <class T>
			static inline auto init_w(Vec<T>& w, size_t w_capacity) {
				w.clear();
				w.resize(w_capacity);
			}

			// Call lambda function to create tuple of RHS expressions
			// Glue w[i] = RHS_i

			// Glue many nodes (equiv: (expr1, expr2, ..., exprn))
			// CALLABLE
			template <class ExprType>
			static inline auto glue_many(ExprType&& expr)
			{
				return std::forward<ExprType>(expr);
			}

			// CALLABLE
			template <class ExprType1, class... ExprType>
			static inline auto glue_many(ExprType1&& expr1, ExprType&&... exprs)
			{
				return (expr1, glue_many(std::forward<ExprType>(exprs)...));
			}

			// w[0] = expr_0, w[1] = expr_1, ...
			template <class T, class Tup, size_t... I>
			static inline auto glue_many(ad::Vec<T>& w, Tup&& tup
				, std::index_sequence<I...>)
			{
				return glue_many((w[I] = std::get<I>(tup))...);
			}

			// glue with vector w, and tuple of expressions
			template <class T, class Tup>
			static inline auto glue_many(ad::Vec<T>& w, Tup&& tup, std::true_type)
			{
				using Tup_pure = std::remove_reference_t<Tup>;
				return glue_many(w, std::forward<Tup>(tup)
					, std::make_index_sequence<std::tuple_size_v<Tup_pure>>());
			}

			// glue with vector w, and one expression
			template <class T, class ExprType>
			static inline auto glue_many(ad::Vec<T>& w, ExprType&& expr, std::false_type)
			{
				auto&& tup = std::make_tuple(std::forward<ExprType>(expr));
				return glue_many(w, std::move(tup));
			}

			// CALLABLE
			// U is either a tuple or expression type
			template <class T, class U>
			static inline auto glue_many(ad::Vec<T>& w, U&& u)
			{
				return glue_many(w, std::forward<U>(u)
					, std::integral_constant<bool, utils::is_tuple<U>::value>());
			}
		};

		// Derived must have the following:
		// expr_eval(Vec<T>& x);
		// init_x(Vec<T>& x, Iter begin, Iter end);
		// Vec<T> x;
		template <class Derived>
		struct ScalarFunctionBaseCRTP
		{
			// CRTP
			inline Derived const& self() const
			{
				return *static_cast<Derived const*>(this);
			}
			inline Derived& self()
			{
				return *static_cast<Derived*>(this);
			}

			// Return expression representing FComposer(FComposed)
			template <class Iter>
			inline auto operator()(Iter begin, Iter end) {
				(this->self()).init_x((this->self()).x, begin, end);
				return (this->self()).expr_eval((this->self()).x);
			}

		};

		// =====================================================================================
		// MODIFY VECTOR FIELD
		// Vector Function Base

		template <class ReturnType, class...ScalarFuncs>
		struct VectorFunctionBase : FunctionBase
		{
			std::tuple<ScalarFuncs...> tup;

			VectorFunctionBase()
				: FunctionBase(sizeof...(ScalarFuncs)), tup()
			{}

			VectorFunctionBase(std::tuple<ScalarFuncs...> const& tup_)
				: FunctionBase(sizeof...(ScalarFuncs))
				, tup(tup_)
			{}

			// Zip returns of each f_i expression
			// Requires begin and end iterator
			template <class Tup, class Iter, size_t... I>
			static inline auto zip_func(Tup&& tup, Iter begin, Iter end, std::index_sequence<I...>) {
				return std::make_tuple(std::get<I>(tup)(begin, end)...);
			}

			template <class Iter, class Tup
				, class = std::enable_if_t<utils::is_tuple<Tup>::value>
			>
				static inline auto zip_func(Tup&& tup, Iter begin, Iter end) {
				using Tup_pure = std::remove_reference_t<Tup>;
				return zip_func(std::forward<Tup>(tup), begin, end
					, std::make_index_sequence<std::tuple_size_v<Tup_pure>>()
				);
			}

			// Zip functions using expr_eval(Vec<T>&)
			// Requires calling expr_eval on Vec<T>
			template <class Tup, class T, size_t...I>
			static inline auto zip_func(Tup&& tup, Vec<T>& x, std::index_sequence<I...>)
			{
				return std::make_tuple(std::get<I>(tup).expr_eval(x)...);
			}

			template <class T, class Tup
				, class = std::enable_if_t<utils::is_tuple<Tup>::value>
			>
				static inline auto zip_func(Tup&& tup, Vec<T>& x)
			{
				using Tup_pure = std::remove_reference_t<Tup>;
				return zip_func(std::forward<Tup>(tup), x
					, std::make_index_sequence<std::tuple_size_v<Tup_pure>>());
			}

			// Returns tuple of return expression from each scalar function
			template <class Iter>
			inline auto operator()(Iter begin, Iter end) {
				return zip_func(this->tup, begin, end);
			}

			// Provide Vec<T>& x as the "x"-vector to create an expression
			template <class T>
			inline auto expr_eval(Vec<T>& x)
			{
				return zip_func(this->tup, x);
			}

		};

		// Vector Field
		// Wrapper of tuple of scalar functions
		// If there are more than threshold number of scalar functions, we use multi-threading
		// Fs... should always be lambda functions
		template <class ReturnType, class... Fs>
		struct Function
			: VectorFunctionBase<ReturnType, Function<ReturnType, Fs>...>
		{
			using base_type = VectorFunctionBase<ReturnType, Function<ReturnType, Fs>...>;

			// Lambda Function
			Function(Fs&&... fs)
				: base_type(std::make_tuple(Function<ReturnType, Fs>(std::forward<Fs>(fs))...))
			{}

			// Other Function objects
			Function(Function<ReturnType, Fs> const&... fs)
				: base_type(std::make_tuple(fs...))
			{}
			Function(Function<ReturnType, Fs>&&... fs)
				: base_type(std::make_tuple(std::move(fs)...))
			{}

		};

		// =====================================================================================

		// Scalar Function (specialization of Vector Field)
		// ScalarFunctionBase<ReturnType> provides data structures and static functions
		// ScalarFunctionBaseCRTP<...> provides operator() using Function impl of expr_eval()
		template <class ReturnType, class F>
		struct Function<ReturnType, F>
			: ScalarFunctionBase<ReturnType>
			, ScalarFunctionBaseCRTP<Function<ReturnType, F>>
		{
			using base_type = ScalarFunctionBase<ReturnType>;

			// Lambda Function
			F f;

			Function(F&& f)
				: f(f)
			{}

			Function(Function const& F_)
				: f(F_.f)
			{}

			// Returns correct expression reprsenting f(x)
			template <class T>
			inline auto expr_eval(Vec<T>& x)
			{
				// Initialize this->w
				constexpr size_t w_capacity = std::tuple_size_v<
					std::result_of_t<F(Vec<ReturnType>&, Vec<ReturnType>&)>
				>;
				this->init_w(this->w, w_capacity);
				// Create expression
				auto&& exprs = this->f(x, this->w);
				return base_type::glue_many(this->w, std::move(exprs));
			}

		};

		// Composition: forward declaration
		template <class ReturnType, class FComposer, class FComposed
			, bool scalar = std::is_base_of_v<
			ScalarFunctionBase<ReturnType>
			, std::remove_reference_t<FComposer> // VERY IMPORTANT TO REMOVE REF
			>
			, class = std::enable_if_t<
			std::is_base_of_v<FunctionBase, std::remove_reference_t<FComposer>> &
			std::is_base_of_v<FunctionBase, std::remove_reference_t<FComposed>>
			>
		>
			struct ComposedFunction;

	} // end core

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

	// Composition Definition
	namespace core {

		// Composition: Vector Function of Vector Function
		// Assumes FComposer and FComposed are derived from FunctionBase
		namespace details {

			// Initialize tuple in ComposedFunction (vector)
			// new tuple element = compose(old tuple element, FComposed)
			template <class ReturnType, class FComposed, class Tup, size_t...I>
			static inline auto init_tup(Tup&& tup, FComposed&& fc
				, std::index_sequence<I...>)
			{
				return std::make_tuple(
					ad::compose(std::get<I>(tup), std::forward<FComposed>(fc))...
				);
			}

			template <class ReturnType, class FComposed, class Tup
				, class = std::enable_if_t<utils::is_tuple<Tup>::value>
			>
				static inline auto init_tup(Tup&& tup, FComposed&& fc)
			{
				using Tup_pure = std::remove_reference_t<Tup>;
				return init_tup<ReturnType>(std::forward<Tup>(tup), std::forward<FComposed>(fc)
					, std::make_index_sequence<std::tuple_size_v<Tup_pure>>());
			}


			// ONLY NEEDED FOR TYPE COMPUTATION
			template <class ReturnType, class...Fs>
			inline auto make_VectorFunctionBase(std::tuple<Fs...> const& tup)
			{
				return VectorFunctionBase<ReturnType, Fs...>(tup);
			}

			template <class ReturnType, class FComposer, class FComposed>
			using VectorFunctionBase_Composed =
				decltype(make_VectorFunctionBase<ReturnType>(
					init_tup<ReturnType>(std::declval<FComposer>().tup, std::declval<FComposed>())
					));

		} // end details 

		template <class ReturnType, class FComposer, class FComposed>
		struct ComposedFunction<ReturnType, FComposer, FComposed, false>
			: details::VectorFunctionBase_Composed<ReturnType, FComposer, FComposed>
		{
			using base_type =
				details::VectorFunctionBase_Composed<ReturnType, FComposer, FComposed>;

			ComposedFunction(FComposer&& fcomposer, FComposed&& fcomposed)
				: base_type(details::init_tup<ReturnType>(
					fcomposer.tup, std::forward<FComposed>(fcomposed)
					))
			{}
		};

		// Composition: Scalar Function of Vector Function
		// ScalarFunctionBase<ReturnType> provides data structures and static functions
		// ScalarFunctionBaseCRTP<...> provides operator() using ComposedFunction impl of expr_eval()
		template <class ReturnType, class FComposer, class FComposed>
		struct ComposedFunction <ReturnType, FComposer, FComposed, true>
			: ScalarFunctionBase<ReturnType>
			, ScalarFunctionBaseCRTP<ComposedFunction<ReturnType, FComposer, FComposed, true>>
		{
			using base_type = ScalarFunctionBase<ReturnType>;

			// Need pure types, otherwise might be references 
			using FComposed_pure = std::remove_reference_t<FComposed>;
			using FComposer_pure = std::remove_reference_t<FComposer>;
			FComposed_pure composed;
			FComposer_pure composer;

			ComposedFunction(FComposer&& fcomposer, FComposed&& fcomposed)
				: composed(std::forward<FComposed>(fcomposed))
                , composer(std::forward<FComposer>(fcomposer))
			{}

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
				auto&& expr_composed_glued = base_type::glue_many(this->w, std::move(expr_composed));
				// Get correct expr from composer function
				auto&& expr_composer = this->composer.expr_eval(this->w);
				// Finally, glue expr_composed_glued, expr_composer
				// Note that expr_composer must be a single expression by specialization
				return base_type::glue_many(std::move(expr_composed_glued)
					, std::move(expr_composer));
			}

		};

		// Helper is_Function
		template <class T>
		struct is_Function
		{
			static constexpr bool value = std::is_base_of_v<FunctionBase,
				std::remove_reference_t<T>>;
		};

	} // namespace core

	namespace core {

		template <class ReturnType, class... Fs>
		struct Make_Function
		{
			static auto make(Fs&&... fs)
			{
				return core::Function<ReturnType, Fs...>(std::forward<Fs>(fs)...);
			}
		};

		// Lvalue
		template <class ReturnType, class... Fs>
		struct Make_Function<ReturnType, Function<ReturnType, Fs>&... >
		{
			static auto make(Function<ReturnType, Fs> const&... fs)
			{
				return core::Function<ReturnType, Fs...>(fs...);
			}
		};

		// Rvalue
		template <class ReturnType, class... Fs>
		struct Make_Function<ReturnType, Function<ReturnType, Fs>... >
		{
			static auto make(Function<ReturnType, Fs>&&... fs)
			{
				return core::Function<ReturnType, Fs...>(std::move(fs)...);
			}
		};
	} // end core

	// Make Function Object with lambda functions
	template <class ReturnType = double, class... Fs>
	inline auto make_function(Fs&&... fs)
	{
		return core::Make_Function<ReturnType, Fs... >::make(std::forward<Fs>(fs)...);
	}


} // end ad
