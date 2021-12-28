defmodule NxMathTest do
  use ExUnit.Case
  doctest NxMath

  @exp_max16 10
  @ratio_epsilon_f16 0.4

  test "exponential f16" do
    ExUnit.configuration()[:seed]
    rand = for(_ <- 1..100, do: :rand.uniform() * (@exp_max16 * 2) - @exp_max16)
    t16 = Nx.tensor(rand, type: {:f, 16})

    assert Nx.less_equal(
             Nx.divide(
               Nx.subtract(
                 NxMath.exp16(t16),
                 Nx.exp(t16) |> Nx.as_type({:f, 16})
               ),
               Nx.exp(t16)
             )
             |> Nx.abs()
             |> Nx.reduce_max(),
             @ratio_epsilon_f16
           ) == Nx.tensor(1, type: {:u, 8})
  end
end
