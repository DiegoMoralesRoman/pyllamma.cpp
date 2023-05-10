#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <tokens.hpp>

namespace boost {
namespace python {

// template <>
// struct to_python_value<const pyllama::Token&> {
//     static PyObject* convert(const pyllama::Token& token) {
//         // Convert the Token to a PyToken instance
//         return incref(object(pyllama::Token(token)).ptr());
//     }
// };

} // namespace python
} // namespace boost
