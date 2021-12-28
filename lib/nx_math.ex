defmodule NxMath do
  import Nx.Defn

  @moduledoc """
  Documentation for `NxMath`.
  """

  @log2 Nx.log(2)

  defnp factorial(x) do
    {factorial, _} =
      while {factorial = 1, x}, Nx.greater(x, 1) do
        {factorial * x, x - 1}
      end

    factorial
  end

  defnp broadcast(s, t, {type, bit}) do
    Nx.broadcast(Nx.tensor(s, type: {type, bit}), Nx.shape(t))
  end

  defnp c(n) do
    if n == 1 do
      1 - Nx.log(2)
    else
      -Nx.power(Nx.log(2), n)
    end
  end

  defnp(f(x, i), do: c(i) / factorial(i) * Nx.power(x, i))

  @doc """
  Calculates the exponential of each element in the tensor.

  ## Examples

      iex> NxMath.exp16(0)
      #Nx.Tensor<
        f16
        1.0
      >
  """
  defn exp16(t) do
    greater_equal_12 = Nx.greater_equal(t, 12)
    less_12 = Nx.less(t, 12)

    t =
      rewrite_types(
        t / @log2,
        max_float_type: {:f, 16}
      )

    xf =
      (t - Nx.floor(t))
      |> Nx.as_type({:f, 16})

    {kxf, _, _} =
      while {kxf = broadcast(0, t, {:f, 16}), n = 1, xf}, Nx.less(n, 4) do
        {
          (kxf + f(xf, n)) |> Nx.as_type({:f, 16}),
          n + 1,
          xf
        }
      end

    is_zero = Nx.equal(t, 0)
    isnt_zero = Nx.not_equal(t, 0)

    value_zero = is_zero * broadcast(1, t, {:f, 16})

    value =
      Nx.round(1024 * (t - kxf + 15) * less_12)
      |> Nx.as_type({:u, 16})

    is_inf = Nx.logical_or(Nx.greater(value, 0x7BFF), greater_equal_12)
    isnt_inf = Nx.logical_or(Nx.greater(value, 0x7BFF), greater_equal_12) |> Nx.logical_not()

    value_not_zero =
      (Nx.logical_and(isnt_inf, isnt_zero) * value)
      |> Nx.as_type({:u, 16})
      |> Nx.bitcast({:f, 16})

    value_not_inf =
      (value_zero + value_not_zero)
      |> Nx.bitcast({:u, 16})

    (value_not_inf + is_inf * 0x7C00)
    |> Nx.as_type({:u, 16})
    |> Nx.bitcast({:f, 16})
  end
end
