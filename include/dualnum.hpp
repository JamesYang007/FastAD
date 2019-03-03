#pragma once

namespace ad {
namespace core {

    // Node data container
    template <class T>
    struct DualNum
    {
        using value_type = T;
        T w, df;   
        DualNum(T w, T df)
            : w(w), df(df)
        {}
    };

} // namepsace core
} // namespace ad 
