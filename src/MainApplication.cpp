#include <tokens/Tensor.hpp>

int main()
{
    char* fname = "/workspaces/VictorNet/data/toyota-corolla.jpg";
    src::tokens::Tensor t{fname};
    t.print_to_image("output.jpg");
}