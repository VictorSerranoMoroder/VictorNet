#include <cstdint>
#include <list>
#include <tokens/Tensor.hpp>

int main()
{
    char* fname = "/workspaces/VictorNet/data/toyota-corolla.jpg";
    std::list<src::tokens::Tensor> l{};
    for (std::size_t i = 0; i < 10000; i++)
    {
        l.emplace_back(fname);
    }

    for (const auto& item : l)
    {
        item.print_to_image("output.jpg");
    }

    l.clear();

    for (std::size_t i = 0; i < 10000; i++)
    {
        l.emplace_back(fname);
    }
}