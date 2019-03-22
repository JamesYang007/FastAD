#include <iterator>

namespace utils {

    // type_traits extension
    //
    // dereferenceable like a pointer

    // Dummy function for return value purpose
    template <typename T>
    auto is_pointer_like()
       -> decltype( * std::declval<T>(), std::true_type{} );

    // Checks if type T is pointer like (dereferenceable)
    template <typename T>
    using is_pointer_like_dereferenceable = decltype(is_pointer_like<T>());

} // end utils
