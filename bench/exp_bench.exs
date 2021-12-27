defmodule RandSet do
  def rand_set(size) do
    for(_ <- 1..size, do: :rand.uniform())
  end

  def make_input(size) do
    rand = rand_set(size)
    t16 = Nx.tensor(rand, type: {:f, 16})
    t32 = Nx.tensor(rand, type: {:f, 32})
    t64 = Nx.tensor(rand, type: {:f, 64})

    %{
      input16: t16,
      input32: t32,
      input64: t64
    }
  end
end

inputs = %{
  "Small" => RandSet.make_input(1_000),
  "Medium" => RandSet.make_input(10_000),
  "Large" => RandSet.make_input(100_000)
}

benches = %{
  "Nx exp f16" => fn %{input16: input} -> Nx.exp(input) end,
  "Nx exp f32" => fn %{input32: input} -> Nx.exp(input) end,
  "Nx exp f64" => fn %{input64: input} -> Nx.exp(input) end,
  "EXLA exp f16" => fn %{input16: input} -> EXLA.jit(&Nx.exp/1, [input]) end,
  "EXLA exp f32" => fn %{input32: input} -> EXLA.jit(&Nx.exp/1, [input]) end,
  "EXLA exp f64" => fn %{input64: input} -> EXLA.jit(&Nx.exp/1, [input]) end
}

Benchee.run(
  benches,
  inputs: inputs,
  time: 10,
  memory_time: 2
)
