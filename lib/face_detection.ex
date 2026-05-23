if ImageVision.ortex_configured?() do
  defmodule Image.FaceDetection do
    @moduledoc """
    Face detection — where are the faces in this image?

    Returns one entry per detected face, each with a bounding box,
    a confidence score, and five facial landmarks (right eye,
    left eye, nose tip, right mouth corner, left mouth corner).

    ## Quick start

        iex> image = Image.open!("./test/support/images/group.jpg")
        iex> [%{box: _, score: _, landmarks: _} | _] = Image.FaceDetection.detect(image)

    ## Default model

    [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
    — the OpenCV team's production face detector. Roughly **340
    KB on disk**, MIT licensed, real-time on CPU. The 2023-March
    export hosted at `opencv/face_detection_yunet` produces
    decoded boxes + keypoints + scores directly.

    Override the default via `:repo` and `:model_file`. The
    output shape this module decodes is the YuNet 2023-March
    convention; `SCRFD` and `BlazeFace` exports differ enough
    that they need a different post-processor.

    ## Drawing detections

    Use `draw_boxes/3` to overlay rectangles + landmarks on the
    original image:

        image
        |> Image.FaceDetection.detect()
        |> Image.FaceDetection.draw_boxes(image)

    ## Face-aware crop

    `crop_largest/2` is a convenience for the common "crop to
    the most prominent face" case (used by CDN parameters like
    ImageKit `z-`, Cloudflare `face-zoom`, and `gravity: :face`):

        {:ok, portrait} = Image.FaceDetection.crop_largest(image, padding: 0.2)

    ## Optional dependency

    This module is only available when [Ortex](https://hex.pm/packages/ortex)
    is configured in your application's `mix.exs`.

    """

    alias Vix.Vips.Image, as: Vimage

    @default_repo "opencv/face_detection_yunet"
    @default_model_file "face_detection_yunet_2023mar.onnx"

    # YuNet's 2023-March ONNX export has a fixed 640×640 input.
    # Older / community exports may differ; configure via the
    # private `@input_size` constant if you swap the model.
    @input_size 640
    @default_min_score 0.6
    @default_nms_iou 0.3

    @typedoc """
    A single detected face.

    * `:box` is `{x, y, width, height}` in pixel coordinates of
      the original image. `(x, y)` is the top-left corner.

    * `:score` is the confidence score, a float in `[0.0, 1.0]`.

    * `:landmarks` is a list of five `{x, y}` tuples in pixel
      coordinates: right eye, left eye, nose tip, right mouth
      corner, left mouth corner — in that order.
    """
    @type face :: %{
            box: {non_neg_integer(), non_neg_integer(), pos_integer(), pos_integer()},
            score: float(),
            landmarks: [{number(), number()}]
          }

    @doc """
    Detects faces in an image and returns a list of detections
    sorted by descending confidence.

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `options` is a keyword list of options.

    ### Options

    * `:min_score` is the minimum confidence score, a float in
      `[0.0, 1.0]`, that a detection must meet to be returned.
      The default is `#{@default_min_score}`.

    * `:nms_iou` is the IoU threshold for non-maximum
      suppression. Detections that overlap more than this
      threshold are collapsed. Lower values keep fewer
      overlapping faces. The default is `#{@default_nms_iou}`.

    * `:repo` is the HuggingFace repository for the YuNet ONNX
      export. Default `"#{@default_repo}"`.

    * `:model_file` is the ONNX filename within the repository.
      Default `"#{@default_model_file}"`.

    ### Returns

    * A list of `t:face/0` maps, sorted by descending `:score`.
      Empty list when no face meets the threshold.

    ### Examples

        iex> image = Image.open!("./test/support/images/group.jpg")
        iex> faces = Image.FaceDetection.detect(image, min_score: 0.7)
        iex> is_list(faces) and Enum.all?(faces, &match?(%{box: _, score: _, landmarks: _}, &1))
        true

    """
    @spec detect(image :: Vimage.t(), options :: Keyword.t()) :: [face()]
    def detect(%Vimage{} = image, options \\ []) do
      min_score = Keyword.get(options, :min_score, @default_min_score)
      nms_iou = Keyword.get(options, :nms_iou, @default_nms_iou)
      repo = Keyword.get(options, :repo, @default_repo)
      model_file = Keyword.get(options, :model_file, @default_model_file)

      # `vips_thumbnail` (used inside `preprocess/1`) auto-applies
      # the EXIF `Orientation` tag, so the model receives the
      # upright image. The rest of this function operates on the
      # same upright frame so `scale_x` / `scale_y` and the box-
      # clamping `max_width` / `max_height` all agree. Without
      # this step, iPhone photos (stored as landscape pixels +
      # `Orientation = 6/8`) detect the face correctly but report
      # box coordinates that, when applied to the un-rotated
      # buffer, land elsewhere in the frame.
      image = Image.autorotate!(image)

      model = load_model(repo, model_file)

      {tensor, scale_x, scale_y} = preprocess(image)

      outputs = Ortex.run(model, tensor)

      outputs
      |> postprocess(scale_x, scale_y,
        min_score: min_score,
        nms_iou: nms_iou,
        max_width: Image.width(image),
        max_height: Image.height(image)
      )
    end

    @doc """
    Returns just the bounding boxes of detected faces, sorted
    by descending confidence. Convenience over `detect/2` when
    landmarks aren't needed.

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `options` is forwarded to `detect/2`.

    ### Returns

    * A list of `{x, y, width, height}` tuples in pixel
      coordinates of the original image.

    """
    @spec boxes(image :: Vimage.t(), options :: Keyword.t()) ::
            [{non_neg_integer(), non_neg_integer(), pos_integer(), pos_integer()}]
    def boxes(%Vimage{} = image, options \\ []) do
      image |> detect(options) |> Enum.map(& &1.box)
    end

    @doc """
    Crops the image to the largest detected face.

    The largest face is chosen by bounding-box area. The crop
    is expanded by `:padding` (a fraction of each face dimension)
    to leave breathing room around the face, then clipped to
    the image bounds. If no face is detected, returns
    `{:error, :no_face_detected}`.

    Used as the wire-in point for face-aware crop bias in
    `image_plug` (`gravity: :face`, ImageKit `z-`,
    Cloudflare `face-zoom`).

    ### Arguments

    * `image` is any `t:Vix.Vips.Image.t/0`.

    * `options` is a keyword list. Detection options
      (`:min_score`, `:nms_iou`, `:input_size`, etc.) are
      forwarded to `detect/2`.

    ### Options

    * `:padding` is a float in `[0.0, 5.0]` controlling how
      much room is kept around the face. `0.0` is a tight crop
      to the bounding box; `0.5` adds 50% on each side; `1.0`
      doubles the bounding box. Default `0.5` — matches the
      Cloudflare Images `face-zoom=0.5` default and tends to
      include shoulders for portrait-style photos.

    ### Returns

    * `{:ok, cropped_image}` or

    * `{:error, :no_face_detected}` if no detection met
      `:min_score`.

    """
    @spec crop_largest(image :: Vimage.t(), options :: Keyword.t()) ::
            {:ok, Vimage.t()} | {:error, :no_face_detected}
    def crop_largest(%Vimage{} = image, options \\ []) do
      padding = Keyword.get(options, :padding, 0.5)

      # Detection runs in the EXIF-rotated frame (see comment in
      # `detect/2`). The crop has to run in the same frame, so we
      # autorotate here too. Cheap no-op when the image has no
      # orientation tag or `Orientation = 1`.
      image = Image.autorotate!(image)
      faces = detect(image, options)

      case largest_face(faces) do
        nil ->
          {:error, :no_face_detected}

        %{box: {x, y, w, h}} ->
          {cx, cy, cw, ch} =
            expand_box(x, y, w, h, padding, Image.width(image), Image.height(image))

          Image.crop(image, cx, cy, cw, ch)
      end
    end

    defp largest_face([]), do: nil

    defp largest_face(faces) do
      Enum.max_by(faces, fn %{box: {_x, _y, w, h}} -> w * h end)
    end

    defp expand_box(x, y, w, h, padding, max_w, max_h) do
      pad_x = round(w * padding)
      pad_y = round(h * padding)
      x1 = max(x - pad_x, 0)
      y1 = max(y - pad_y, 0)
      x2 = min(x + w + pad_x, max_w)
      y2 = min(y + h + pad_y, max_h)
      {x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)}
    end

    @doc """
    Draws bounding boxes and the five facial landmarks for each
    detection onto an image.

    Builds an SVG overlay (one box + five dots per face) and
    composites it onto the image. The score is rendered as a
    percentage label above each box.

    ### Arguments

    * `detections` is the list returned from `detect/2`.

    * `image` is the image upon which detection was run.

    * `options` is a keyword list of options.

    ### Options

    * `:color` is the CSS colour used for boxes and landmarks.
      Default `"#3cb44b"` (a high-contrast green).

    * `:stroke_width` is the bounding-box stroke width in
      pixels. Default `2`.

    * `:landmark_radius` is the radius of each landmark dot in
      pixels. Default `3`.

    * `:font_size` is the score-label text size in pixels.
      Default `13`.

    * `:show_landmarks?` when `false` skips drawing the five
      landmark dots. Default `true`.

    ### Returns

    * The annotated `t:Vix.Vips.Image.t/0`.

    """
    @spec draw_boxes([face()], Vimage.t(), Keyword.t()) :: Vimage.t()
    def draw_boxes(detections, %Vimage{} = image, options \\ []) do
      color = Keyword.get(options, :color, "#3cb44b")
      stroke_width = Keyword.get(options, :stroke_width, 2)
      radius = Keyword.get(options, :landmark_radius, 3)
      font_size = Keyword.get(options, :font_size, 13)
      show_landmarks? = Keyword.get(options, :show_landmarks?, true)

      width = Image.width(image)
      height = Image.height(image)

      label_height = font_size + 5
      text_baseline = font_size + 1

      svg_body =
        Enum.map(detections, fn %{box: {x, y, w, h}, score: score, landmarks: marks} ->
          text = "#{Float.round(score * 100, 1)}%"
          label_y = max(0, y - label_height)
          label_w = round(String.length(text) * font_size * 0.55) + 8

          landmark_svg =
            if show_landmarks? do
              Enum.map(marks, fn {mx, my} ->
                ~s|<circle cx="#{mx}" cy="#{my}" r="#{radius}" fill="#{color}"/>|
              end)
              |> Enum.join()
            else
              ""
            end

          """
          <rect x="#{x}" y="#{y}" width="#{w}" height="#{h}"
                fill="none" stroke="#{color}" stroke-width="#{stroke_width}"/>
          <rect x="#{x}" y="#{label_y}" width="#{label_w}" height="#{label_height}"
                fill="#{color}"/>
          <text x="#{x + 4}" y="#{label_y + text_baseline}"
                font-family="sans-serif" font-size="#{font_size}" font-weight="bold" fill="white">#{text}</text>
          #{landmark_svg}
          """
        end)
        |> Enum.join()

      svg = """
      <svg xmlns="http://www.w3.org/2000/svg" width="#{width}" height="#{height}">
        #{svg_body}
      </svg>
      """

      {:ok, overlay} = Image.open(svg, access: :sequential)
      Image.compose!(image, overlay)
    end

    # --- Internal: model loading + preprocessing + post-processing ---

    defp load_model(repo, model_file) do
      key = {__MODULE__, repo, model_file}

      case :persistent_term.get(key, nil) do
        nil ->
          path = ImageVision.ModelCache.fetch!(repo, model_file)
          model = Ortex.load(path)
          :persistent_term.put(key, model)
          model

        model ->
          model
      end
    end

    # Resize to @input_size × @input_size, transpose to NCHW,
    # and add a batch axis. YuNet expects raw `[0, 255]` u8
    # values cast to f32 — no normalisation. Returns the
    # tensor plus the per-axis scale factor needed to map
    # predictions back to the original image.
    defp preprocess(%Vimage{} = image) do
      flat =
        image
        |> Image.flatten!()
        |> Image.to_colorspace!(:srgb)

      {:ok, resized} =
        Vix.Vips.Operation.thumbnail_image(flat, @input_size,
          height: @input_size,
          size: :VIPS_SIZE_FORCE
        )

      tensor =
        resized
        |> Image.to_nx!(backend: Nx.BinaryBackend)
        |> Nx.as_type(:f32)
        |> Nx.transpose(axes: [2, 0, 1])
        |> Nx.new_axis(0)

      scale_x = Image.width(image) / @input_size
      scale_y = Image.height(image) / @input_size

      {tensor, scale_x, scale_y}
    end

    # YuNet 2023-March emits 12 output tensors in a fixed
    # order: `cls_8, cls_16, cls_32, obj_8, obj_16, obj_32,
    # bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32`.
    #
    # Per stride we have:
    #   * `cls_*` — face classification logits, shape `{1, hw, 1}`
    #   * `obj_*` — objectness logits, shape `{1, hw, 1}`
    #   * `bbox_*` — bbox regression deltas, shape `{1, hw, 4}`
    #     (interpreted as `cx-offset, cy-offset, log_w, log_h`
    #     in stride units)
    #   * `kps_*` — five `(x, y)` keypoint deltas, shape
    #     `{1, hw, 10}` in stride units
    #
    # The score for each anchor is `sigmoid(cls) × sigmoid(obj)`.
    defp postprocess(outputs, scale_x, scale_y, options) do
      min_score = Keyword.fetch!(options, :min_score)
      nms_iou = Keyword.fetch!(options, :nms_iou)
      max_width = Keyword.fetch!(options, :max_width)
      max_height = Keyword.fetch!(options, :max_height)

      tensors =
        outputs
        |> Tuple.to_list()
        |> Enum.map(&Nx.backend_transfer(&1, Nx.BinaryBackend))

      [cls8, cls16, cls32, obj8, obj16, obj32, bbox8, bbox16, bbox32, kps8, kps16, kps32] =
        tensors

      stride_inputs = [
        {8, cls8, obj8, bbox8, kps8},
        {16, cls16, obj16, bbox16, kps16},
        {32, cls32, obj32, bbox32, kps32}
      ]

      stride_inputs
      |> Enum.flat_map(fn {stride, cls, obj, bbox, kps} ->
        decode_stride(cls, obj, bbox, kps,
          stride: stride,
          scale_x: scale_x,
          scale_y: scale_y,
          min_score: min_score,
          max_width: max_width,
          max_height: max_height
        )
      end)
      |> nms(nms_iou)
      |> Enum.sort_by(& &1.score, :desc)
    end

    # For each anchor in the feature map, generate a prior
    # `(cx, cy)` at the centre of its grid cell, decode bbox +
    # keypoints from the regression deltas, and emit a
    # detection if `cls × obj` exceeds the score threshold.
    defp decode_stride(cls, obj, bbox, kps, options) do
      stride = Keyword.fetch!(options, :stride)
      scale_x = Keyword.fetch!(options, :scale_x)
      scale_y = Keyword.fetch!(options, :scale_y)
      min_score = Keyword.fetch!(options, :min_score)
      max_width = Keyword.fetch!(options, :max_width)
      max_height = Keyword.fetch!(options, :max_height)

      grid_size = div(@input_size, stride)
      hw = grid_size * grid_size

      cls_list = cls |> Nx.reshape({hw}) |> Nx.to_flat_list()
      obj_list = obj |> Nx.reshape({hw}) |> Nx.to_flat_list()
      bbox_list = bbox |> Nx.reshape({hw, 4}) |> Nx.to_list()
      kps_list = kps |> Nx.reshape({hw, 10}) |> Nx.to_list()

      Enum.zip([0..(hw - 1), cls_list, obj_list, bbox_list, kps_list])
      |> Enum.flat_map(fn {idx, c, o, [bx, by, bw, bh], k} ->
        # YuNet's `cls` and `obj` outputs are already
        # post-sigmoid probabilities in `[0, 1]`. The combined
        # score is the geometric mean of the two — same
        # formula OpenCV's `cv::FaceDetectorYN` uses.
        score = :math.sqrt(max(c * o, 0.0))

        if score >= min_score do
          row = div(idx, grid_size)
          col = rem(idx, grid_size)
          # YuNet prior centre is at the top-left corner of the
          # cell, in stride units.
          prior_cx = col * stride
          prior_cy = row * stride

          cx = (bx + col) * stride
          cy = (by + row) * stride
          w = :math.exp(bw) * stride
          h = :math.exp(bh) * stride

          x1 = max((cx - w / 2) * scale_x, 0)
          y1 = max((cy - h / 2) * scale_y, 0)
          x2 = min((cx + w / 2) * scale_x, max_width)
          y2 = min((cy + h / 2) * scale_y, max_height)

          width = x2 - x1
          height = y2 - y1

          if width > 0 and height > 0 do
            landmarks = decode_landmarks(k, prior_cx, prior_cy, stride, scale_x, scale_y)

            [
              %{
                box: {round(x1), round(y1), round(width), round(height)},
                score: score,
                landmarks: landmarks
              }
            ]
          else
            []
          end
        else
          []
        end
      end)
    end

    # Five (x, y) pairs decoded as `prior_centre + delta * stride`.
    defp decode_landmarks(deltas, prior_cx, prior_cy, stride, scale_x, scale_y) do
      deltas
      |> Enum.chunk_every(2)
      |> Enum.map(fn [dx, dy] ->
        x = (dx * stride + prior_cx) * scale_x
        y = (dy * stride + prior_cy) * scale_y
        {Float.round(x, 1), Float.round(y, 1)}
      end)
    end

    # Greedy NMS in pure Elixir. Sort by score, repeatedly pop
    # the highest-scoring detection and drop everything whose
    # IoU with it exceeds the threshold.
    defp nms(detections, iou_threshold) do
      detections
      |> Enum.sort_by(& &1.score, :desc)
      |> nms_loop([], iou_threshold)
    end

    defp nms_loop([], kept, _iou), do: Enum.reverse(kept)

    defp nms_loop([head | rest], kept, iou_threshold) do
      survivors = Enum.reject(rest, fn d -> iou(head.box, d.box) >= iou_threshold end)
      nms_loop(survivors, [head | kept], iou_threshold)
    end

    defp iou({ax, ay, aw, ah}, {bx, by, bw, bh}) do
      ax2 = ax + aw
      ay2 = ay + ah
      bx2 = bx + bw
      by2 = by + bh

      ix1 = max(ax, bx)
      iy1 = max(ay, by)
      ix2 = min(ax2, bx2)
      iy2 = min(ay2, by2)

      iw = max(0, ix2 - ix1)
      ih = max(0, iy2 - iy1)
      intersection = iw * ih
      union = aw * ah + bw * bh - intersection

      if union > 0, do: intersection / union, else: 0.0
    end
  end
end
