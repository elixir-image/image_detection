# Image Detection

Image Detection is an [Image](https://hex.pm/packages/image)-based object detection library based upon the [YOLO V8](https://docs.ultralytics.com) ML model.

The documentation can be found at [https://hexdocs.pm/image_detection](https://hexdocs.pm/image_detection).

## Installation

`Image Detection` can be installed by adding `image_detection` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:image_detection_, "~> 0.1.0"}
  ]
end
```

## Configuration

THIS IS AN EXPERIMENTAL LIBRARY. Please do not use in production. Testing is not yet complete.

The code is an adaptation of the livecoding demo by Hans Elias (@hansihe) at his [talk](https://www.youtube.com/watch?v=OsxGB6MbA8o&t=1s) at the Elixir Warsaw meetup in March 2023.

## Accessing the Yolo V8 model

The YOLO v8 model is GPL 3.0 licensed and must be built separately. Python 3.10 is required (apparently it won't work with 3.11).

```bash
% pip3.10 install ultralytics
% pip3.10 install onnx
% yolo export model=yolov8n.pt format=onnx imgsz=640
```

The "n" model is the smallest - we can maybe tolerate a larger one. And we need to find way to host the model or download the `.onnx` from somewhere.

## References

* The original talk by @hansihe is at https://www.youtube.com/watch?v=OsxGB6MbA8o&t=1s
* The original models are stored at https://github.com/ultralytics/assets/releases
* Learning: https://learnopencv.com/ultralytics-yolov8/
* Exploration: https://medium.com/mlearning-ai/yolo-v8-the-real-state-of-the-art-eda6c86a1b90

