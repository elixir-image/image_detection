defmodule ImageVision.MixProject do
  use Mix.Project

  @version "0.4.0"
  @app_name "image_vision"

  def project do
    [
      app: String.to_atom(@app_name),
      version: @version,
      elixir: "~> 1.17",
      deps: deps(),
      elixirc_paths: elixirc_paths(Mix.env()),
      source_url: "https://github.com/elixir-image/image_vision",
      docs: docs(),
      build_embedded: Mix.env() == :prod,
      start_permanent: Mix.env() == :prod,
      description: description(),
      package: package(),
      aliases: aliases(),
      dialyzer: [
        ignore_warnings: ".dialyzer_ignore_warnings",
        plt_add_apps: ~w(mix nx ortex bumblebee)a
      ],
      compilers: Mix.compilers()
    ]
  end

  defp description do
    """
    Simple image classification, embedding, object detection,
    segmentation, background removal, captioning, and zero-shot
    classification for the `image` library. Powered by Bumblebee and
    Ortex with strong, permissively-licensed defaults — no ML
    expertise required.
    """
  end

  def application do
    [
      mod: {ImageVision.Application, []},
      extra_applications: [:logger, :inets, :crypto]
    ]
  end

  defp deps do
    [
      # The core image-processing library.
      {:image, "~> 0.66"},

      # Used to download ONNX model weights on first call.
      {:req, "~> 0.5"},

      # Transitive workaround: the `:color` dep (pulled in via `:image`)
      # has a `Code.ensure_loaded?(Plug.Router)` check that misbehaves on
      # Elixir 1.20-rc.4. Including `:plug` ensures the conditional
      # branch compiles cleanly. Remove once `:color` upstream is fixed.
      {:plug, "~> 1.15", optional: true},

      # --- ML deps ---
      #
      # Segmentation and detection run ONNX models via Ortex.
      {:ortex, "~> 0.1", optional: true},
      #
      # Classification and embedding use Bumblebee servings.
      {:bumblebee, "~> 0.7", optional: true},
      #
      # Nx and EXLA are required for inference.
      {:nx, "~> 0.12"},
      {:exla, "~> 0.12"},

      # --- Tooling ---
      {:ex_doc, "~> 0.18", only: [:release, :dev, :docs]},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ] ++ maybe_json_polyfill()
  end

  defp maybe_json_polyfill do
    if Code.ensure_loaded?(:json) do
      []
    else
      [{:json_polyfill, "~> 0.2 or ~> 1.0"}]
    end
  end

  defp package do
    [
      maintainers: ["Kip Cole"],
      licenses: ["Apache-2.0"],
      links: links(),
      files: [
        "lib",
        "guides",
        "logo.jpg",
        "mix.exs",
        "README*",
        "CHANGELOG*",
        "LICENSE*"
      ]
    ]
  end

  def links do
    %{
      "GitHub" => "https://github.com/elixir-image/image_vision",
      "Readme" => "https://github.com/elixir-image/image_vision/blob/v#{@version}/README.md",
      "Changelog" =>
        "https://github.com/elixir-image/image_vision/blob/v#{@version}/CHANGELOG.md",
      "image" => "https://hex.pm/packages/image",
      "libvips" => "https://www.libvips.org"
    }
  end

  def docs do
    [
      source_ref: "v#{@version}",
      main: "readme",
      logo: "logo.jpg",
      extra_section: "Guides",
      extras: extras(),
      groups_for_extras: groups_for_extras(),
      formatters: ["html", "markdown"],
      skip_undefined_reference_warnings_on: ["changelog", "CHANGELOG.md"]
    ]
  end

  defp groups_for_extras do
    [
      Guides: ~r"guides/.*"
    ]
  end

  defp extras do
    Enum.filter(
      [
        "README.md",
        "LICENSE.md",
        "CHANGELOG.md",
        "guides/overview.md",
        "guides/classification.md",
        "guides/segmentation.md",
        "guides/detection.md",
        "guides/face_detection.md",
        "guides/background.md",
        "guides/captioning.md",
        "guides/zero_shot.md"
      ],
      &File.exists?/1
    )
  end

  def aliases do
    []
  end

  defp elixirc_paths(:test), do: ["lib", "test"]
  defp elixirc_paths(_), do: ["lib"]
end
