#pragma once

namespace ad {
namespace core {

    // Node data container
    template <class T>
    struct DualNum
    {
        using valuetype = T;
        T w, df;   
        DualNum(T w, T df)
            : w(w), df(df)
        {}
    };

} // namepsace core
} // namespace ad 
