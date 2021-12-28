defmodule NxMath.MixProject do
  use Mix.Project

  @source_url "https://github.com/zacky1972/nx_math"
  @version "0.1.0-dev"

  def project do
    [
      app: :nx_math,
      version: @version,
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
      {:ex_doc, "~> 0.26", only: :dev, runtime: false},
      {:dialyxir, "~> 1.1", only: :dev, runtime: false},
      {:git_hooks, "~> 0.6.4", only: :dev, runtime: false},
      {:benchee, "~> 1.0", only: :dev},
      {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true}
    ]
  end

  defp docs do
    [
      main: "NxMath",
      source_ref: "v#{@version}",
      source_url: @source_url
    ]
  end
end
