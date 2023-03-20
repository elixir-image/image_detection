defmodule Image.Detection do
  @moduledoc """
  Object detection based upon the [YOLO V8](https://docs.ultralytics.com)
  ML model.

  THIS IS AN EXPERIMENTAL MODULE. Please do not use in production. Testing
  is not yet complete.

  The code is an adaptation of the livecoding demo by Hans Elias (@hansihe) at
  his [talk](https://www.youtube.com/watch?v=OsxGB6MbA8o&t=1s) at the
  Elixir Warsaw meetup in March 2023.

  ### Accessing the Yolo V8 model

  The YOLO v8 model is GPL 3.0 licensed and must be built separately. Python
  3.10 is required (apparently it won't work with 3.11).

  ```bash
  % pip3.10 install ultralytics
  % pip3.10 install onnx
  % yolo export model=yolov8n.pt format=onnx imgsz=640
  ```

  The "n" model is the smallest - we can maybe tolerate a larger one.
  And we need to find way to host the model or download the .onnx from
  somewhere.

  ### Dependencies

  * The dependencies for this module include some dependencies from github
    for now, including forks of `axon_onnx` and @hansihe's `yolov8_elixir`.
    Therefore this module can only be used by cloning the repo and checking out
    the `detection` branch. It can be configured by:

  ```elixir
  def deps do
    [
      {:image, github: "elixir-image/image", branch: "detect"},
      {:kino, "~> 0.9"},
      {:exla, "~> 0.5"},
      {:axon_onnx, github: "elixir-image/axon_onnx"}
    ]
  end
  ```

  ### References

  * The original talk by @hansihe is at https://www.youtube.com/watch?v=OsxGB6MbA8o&t=1s
  * The original models are stored at https://github.com/ultralytics/assets/releases
  * Learning: https://learnopencv.com/ultralytics-yolov8/
  & Exploration: https://medium.com/mlearning-ai/yolo-v8-the-real-state-of-the-art-eda6c86a1b90

  """

  alias Vix.Vips.Image, as: Vimage

  @yolo_model_image_size 640

  @classes :image_detection
           |> Application.app_dir("priv/models/")
           |> Path.join("coco_classes.txt")
           |> File.read!
           |> String.split("\n")
           |> Enum.map(&String.trim/1)

  @type class_name :: String.t

  @typedoc """
  Each bounding box looks like this after
  detection.

  ```elixir
  [
    x :: non_neg_integer(),
    y :: non_neg_integer(),
    width :: integer(),
    height :: integer(),
    confidence :: [float()]
  ]
  ```

  """
  @type bounding_box :: list()

  @type bounding_boxes :: [bounding_box(), ...]

  @doc """
  Detect objects in an image using the yolov8n
  model.

  ### Arguments

  * `image` is any `t:Vimage.t/0` that might be
    returned by `Image.open/2`, `Image.from_kino/1` or
    `Image.from_binary/1`.

  * `model_path` is the path to a Yolo `.onnx` model
    file. The default is "priv/models/yolov8n.onnx".
    Note that this model must be user-provided. See the
    instructions in the `Image.Detection` module docs.

  ### Returns

  * `{:ok, map_of_labels_and_bounding_boxes}` or

  * `{:error, reason}`

  """

  @spec detect(image :: Vimage.t(), model_path: Path.t()) ::
    %{class_name() => bounding_boxes()}

  def detect(%Vimage{} = image, model_path \\ default_model_path()) do
    # Import the model and extract the
    # prediction function and its parameters.
    {model, params} = AxonOnnx.import(model_path)
    {_init_fn, predict_fn} = Axon.build(model, compiler: EXLA)

    # Flatten out any alpha band then resize the image
    # so the longest edge is the same as the model size.
    resized_image =
      image
      |> Image.flatten!()
      |> Image.thumbnail!(@yolo_model_image_size)

    # Then add a black border to expand the shorter dimension
    # so the overall image conforms to the model requirements.
    # The base image will be embedded at `(0,0)` on the
    # canvas so that the bounding box dimensions will still
    # reflect their positions on the original image after
    # re-scaling.
    prepared_image =
      resized_image
      |> Image.embed!(@yolo_model_image_size, @yolo_model_image_size)

    # Move the image to Nx. This is nothing more
    # than moving a pointer under the covers
    # so its efficient. Then conform the data to
    # the shape and type required for the model.
    # Last we add an additional axis that represents
    # the batch (we use only a batch of 1).
    batch =
      prepared_image
      |> Image.to_nx!()
      |> Nx.transpose(axes: [2, 0, 1])
      |> Nx.as_type(:f32)
      |> Nx.divide(255)
      |> Nx.new_axis(0)

    # Run the prediction model, extract the only batch that was sent
    # and transpose the axis back to {width, height} layout for further
    # image processing. Then filter by prediction confidence and rescale
    # the boxes back to their size and position on the original image.
    bounding_boxes =
      predict_fn.(params, batch)[0]
      |> Nx.transpose(axes: [1, 0])
      |> nms(0.5)
      |> rescale_bounding_boxes(resized_image, image)

    # Last step is to zip the class names with the bounding
    # boxes and make a map.
    classes()
    |> Enum.zip(bounding_boxes)
    |> Map.new
  end

  @doc """
  Draws bounding boxes with labels.

  ### Arguments

  * `bounding_boxes` is the result returned from
    Image.Detection.detect/2`.

  * `image` is the image upon which object detection
    was run.

  ### Returns

  * `{:ok, image}` or

  * `{:error, reason}`

  """

  def draw_bbox_with_labels(%{} = bounding_boxes, %Vimage{} = image) do
    {width, height, _bands} = Image.shape(image)

    Enum.reduce(bounding_boxes, image, fn {class_name, boxes}, image ->
      Enum.reduce(boxes, image, fn [x, y, w, h | _probs], image ->
        {:ok, bounding_box_image} =
          Image.Shape.rect(w, h, stroke_color: :red, stroke_width: 4)

        {:ok, text_image} =
          Image.Text.text(class_name, text_fill_color: :red, font_size: 20)

        image
        |> Image.compose!(bounding_box_image, x: x, y: y)
        |> Image.compose!(text_image,
          x: min(max(x, 0), width) + 5,
          y: min(max(y, 0), height) + 5
        )
      end)
    end)
  end

  @doc """
  Filters the bounding boxes returning from
  `Image.Detection.detect/2` returning only those
  which meet a determined confidence threshold.

  ### Arguments

  * `bounding_boxes` is the result returned from
    Image.Detection.detect/2`.

  * `threshold` is a float betwee `0.0` and `1.0`
    below which bounding boxes will be rejected.

  ### Returns

  * `filtered_bounding_boxes`

  ### Note

  * This function is not used in the current detection
    implementation. See `Yolo.NMS.nms/2`.

  """

  def filter_predictions(bboxes, thresh \\ 0.5) do
    boxes = Nx.slice(bboxes, [0, 0], [8400, 4])
    probs = Nx.slice(bboxes, [0, 4], [8400, 80])
    max_prob = Nx.reduce_max(probs, axes: [1])

    sorted_idxs =
      Nx.argsort(max_prob, direction: :desc)

    boxes =
      [boxes, Nx.new_axis(max_prob, 1)]
      |> Nx.concatenate(axis: 1)
      |> Nx.take(sorted_idxs)

    boxes
    |> Nx.to_list()
    |> Enum.take_while(fn [_, _, _, _, prob] -> prob > thresh end)
  end

  defp default_model_path do
    path =
      :image_detection
      |> Application.app_dir("priv/models/")
      |> Path.join("yolov8n.onnx")

    if File.exists?(path) do
      path
    else
      raise ArgumentError,
        """
        The default model was not found at #{inspect path}.
        To install the model ensure python3.10 is installed
        then:

        % pip3.10 install ultralytics
        % pip3.10 install onnyx
        % yolo export model=yolov8n.pt format=onnx imgsz=640

        And move the resulting yolov8n.onnx file to the
        #{inspect path}.
        """
    end
  end

  # Scale the bounding boxes to the size of the original
  # image. Also calculate the top left of the box from
  # the centre points we are given.

  # This should be reimplemented in Nx calls.

  defp rescale_bounding_boxes(bounding_boxes, resized_image, image) do
    x_factor = Image.width(image) / Image.width(resized_image)
    y_factor = Image.height(image) / Image.height(resized_image)

    Enum.map(bounding_boxes, fn boxes ->
      Enum.map(boxes, fn [cx, cy, w, h, confidences] ->
        [
          round((cx - w / 2) * x_factor),
          round((cy - h / 2) * y_factor),
          round(w * x_factor),
          round(h * y_factor),
          confidences
        ]
      end)
    end)
  end

  def classes do
    @classes
  end

  # nms from @hansihe's yolov8_elixir library
  # defp nms(boxes, prob_thresh \\ 0.8, iou_thresh \\ 0.8) do
  defp nms(boxes, prob_thresh, iou_thresh \\ 0.8) do
    {_anchors, data} = Nx.shape(boxes)

    (0..(data - 5))
    |> Enum.map(fn idx ->
      probs =
        boxes
        |> Nx.slice_along_axis(4 + idx, 1, axis: 1)
        |> Nx.reshape({:auto})

      argsort = Nx.argsort(probs, direction: :desc)

      boxes_ordered = Nx.take(Nx.slice_along_axis(boxes, 0, 4, axis: 1), argsort)
      probs_ordered = Nx.new_axis(Nx.take(probs, argsort), 1)

      concated = Nx.concatenate([boxes_ordered, probs_ordered], axis: 1)

      above_thresh =
        concated
        |> Nx.to_batched(1)
        |> Stream.map(&Nx.to_flat_list/1)
        |> Enum.take_while(fn [_, _, _, _, prob] -> prob > prob_thresh end)

      do_nms(above_thresh, [], iou_thresh)
    end)
  end

  defp do_nms([], results, _iou_thresh), do: results

  defp do_nms([box1 | rest], results, iou_thresh) do
    rest =
      rest
      |> Stream.map(fn box2 -> {box2, iou(box1, box2)} end)
      |> Stream.reject(fn {_box2, iou} -> iou > iou_thresh end)
      |> Enum.map(fn {bbox2, _iou} -> bbox2 end)

    do_nms(rest, [box1 | results], iou_thresh)
  end

  defp iou([x1, y1, w1, h1 | _], [x2, y2, w2, h2 | _]) do
    area1 = w1 * h1
    area2 = w2 * h2

    xx = max(x1 - (w1 / 2), x2 - (w2 / 2))
    yy = max(y1 - (h1 / 2), y2 - (h2 / 2))
    aa = min(x1 + (w1 / 2), x2 + (w2 / 2))
    bb = min(y1 + (h2 / 2), y2 + (h2 / 2))

    w = max(0, aa - xx)
    h = max(0, bb - yy)

    intersection_area = w * h

    union_area = area1 + area2 - intersection_area

    intersection_area / union_area
  end
end
