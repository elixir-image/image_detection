defmodule ImageDetection.MixProject do
  use Mix.Project

  @version "0.1.0"
  @app_name "image_detection"

  def project do
    [
      app: String.to_atom(@app_name),
      version: @version,
      elixir: "~> 1.11",
      deps: deps(),
      elixirc_paths: elixirc_paths(Mix.env()),
      source_url: "https://github.com/elixir-image/image_detection",
      docs: docs(),
      build_embedded: Mix.env() == :prod,
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      package: package(),
      aliases: aliases(),
      elixirc_paths: elixirc_paths(Mix.env()),
      preferred_cli_env: preferred_cli_env(),
      dialyzer: [
        ignore_warnings: ".dialyzer_ignore_warnings",
        plt_add_apps: ~w(mix nx plug evision bumblebee)a
      ],
      compilers: Mix.compilers()
    ]
  end

  defp description do
    """
    Image library based object detection using the Yolo V8 model..
    """
  end

  def application do
    [
      extra_applications: [:logger, :inets, :crypto]
    ]
  end

  defp deps do
    [
      {:image, "~> 0.27.0"},
      {:nx, "~> 0.5"},
      {:exla, "~> 0.5"},
      {:axon_onnx, github: "elixir-image/axon_onnx"}
    ]
  end

  defp package do
    [
      maintainers: ["Kip Cole"],
      licenses: ["Apache-2.0"],
      links: links(),
      files: [
        "lib",
        "priv",
        "mix.exs",
        "README*",
        "CHANGELOG*",
        "LICENSE*"
      ]
    ]
  end

  def links do
    %{
      "GitHub" => "https://github.com/image/image_detection",
      "Readme" => "https://github.com/image/image_detection/blob/v#{@version}/README.md",
      "Changelog" => "https://github.com/image/image_detection/blob/v#{@version}/CHANGELOG.md",
      "Vix" => "https://github.com/akash-akya/vix",
      "libvips" => "https://www.libvips.org",
      "eVision (OpenCV)" => "https://github.com/cocoa-xu/evision"
    }
  end

  def docs do
    [
      source_ref: "v#{@version}",
      main: "readme",
      logo: "logo.jpg",
      extra_section: "Guides",
      extras: [
        "README.md",
        "LICENSE.md",
        "CHANGELOG.md",
        "guides/examples.md"
      ],
      formatters: ["html"],
      skip_undefined_reference_warnings_on: ["changelog", "CHANGELOG.md"]
    ]
  end

  defp preferred_cli_env() do
    []
  end

  def aliases do
    []
  end

  defp elixirc_paths(:test), do: ["lib", "src", "mix", "test"]
  defp elixirc_paths(:dev), do: ["lib", "src", "mix", "bench"]
  defp elixirc_paths(:release), do: ["lib", "src"]
  defp elixirc_paths(_), do: ["lib", "src"]
end
