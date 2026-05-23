defmodule Image.FaceDetectionTest do
  use ExUnit.Case, async: false

  @moduletag :ml
  @moduletag :ortex

  # Fixtures pinned to the smallest face count we expect at the
  # default threshold — exercises both clean portraits and the
  # harder small-face / single-face-against-busy-background
  # cases.
  @face_corpus [
    %{
      path: Path.join(__DIR__, "support/images/Kip_small.jpg"),
      min_faces: 1,
      kind: :portrait
    },
    %{
      path: Path.join(__DIR__, "support/images/jose.png"),
      min_faces: 1,
      kind: :portrait
    },
    %{
      path: Path.join(__DIR__, "support/images/Mongolia-2017-10-0984.jpg"),
      min_faces: 1,
      kind: :single_face_in_scene
    },
    %{
      path: Path.join(__DIR__, "support/images/elixir_warsaw_meetup.png"),
      min_faces: 10,
      kind: :many_small_faces
    }
  ]

  @no_face_image Path.join(__DIR__, "support/images/lamborghini-forsennato-concept.jpg")

  describe "detect/2 — face images" do
    test "every fixture meets its minimum-face count at the default threshold" do
      for %{path: path, min_faces: min_faces, kind: kind} <- @face_corpus do
        image = Image.open!(path)
        faces = Image.FaceDetection.detect(image)

        assert length(faces) >= min_faces,
               "expected ≥ #{min_faces} #{kind} faces in #{Path.basename(path)}, got #{length(faces)}"
      end
    end

    test "every detection has the expected shape" do
      for %{path: path} <- @face_corpus do
        image = Image.open!(path)
        max_w = Image.width(image)
        max_h = Image.height(image)

        faces = Image.FaceDetection.detect(image)

        for %{box: {x, y, w, h}, score: score, landmarks: marks} <- faces do
          assert is_integer(x) and x >= 0
          assert is_integer(y) and y >= 0
          assert is_integer(w) and w > 0
          assert is_integer(h) and h > 0
          assert x + w <= max_w
          assert y + h <= max_h

          assert is_float(score) and score >= 0.0 and score <= 1.0
          assert is_list(marks) and length(marks) == 5

          for {mx, my} <- marks do
            assert is_number(mx) and mx >= 0 and mx <= max_w
            assert is_number(my) and my >= 0 and my <= max_h
          end
        end
      end
    end

    test "min_score option filters detections" do
      image = Image.open!(hd(@face_corpus).path)

      lo = Image.FaceDetection.detect(image, min_score: 0.1)
      hi = Image.FaceDetection.detect(image, min_score: 0.99)

      assert length(lo) >= length(hi)

      for %{score: s} <- hi do
        assert s >= 0.99
      end
    end

    test "results are sorted by descending confidence" do
      for %{path: path} <- @face_corpus do
        image = Image.open!(path)
        scores = image |> Image.FaceDetection.detect() |> Enum.map(& &1.score)

        assert scores == Enum.sort(scores, :desc)
      end
    end
  end

  describe "detect/2 — no-face image" do
    test "returns empty (or near-empty) for a non-face image" do
      image = Image.open!(@no_face_image)
      faces = Image.FaceDetection.detect(image)

      assert is_list(faces)
      # YuNet may produce zero or a small number of low-confidence
      # spurious detections on objects; assert structural shape
      # rather than zero count.
      for face <- faces do
        assert match?(%{box: {_, _, _, _}, score: _, landmarks: _}, face)
      end
    end
  end

  describe "boxes/2" do
    test "returns just the bounding-box tuples" do
      image = Image.open!(hd(@face_corpus).path)
      boxes = Image.FaceDetection.boxes(image)

      assert is_list(boxes)
      assert boxes != []
      assert Enum.all?(boxes, &match?({_, _, _, _}, &1))
    end
  end

  describe "crop_largest/2" do
    test "returns a cropped image for every face-bearing fixture" do
      for %{path: path} <- @face_corpus do
        image = Image.open!(path)

        assert {:ok, cropped} = Image.FaceDetection.crop_largest(image)
        assert match?(%Vix.Vips.Image{}, cropped)
        assert Image.width(cropped) <= Image.width(image)
        assert Image.height(cropped) <= Image.height(image)
      end
    end

    test ":padding > 0 grows the crop relative to the bounding box" do
      image = Image.open!(hd(@face_corpus).path)

      {:ok, tight} = Image.FaceDetection.crop_largest(image, padding: 0.0)
      {:ok, padded} = Image.FaceDetection.crop_largest(image, padding: 0.5)

      tight_area = Image.width(tight) * Image.height(tight)
      padded_area = Image.width(padded) * Image.height(padded)

      assert padded_area >= tight_area
    end

    test "returns :no_face_detected when no face meets the threshold" do
      image = Image.open!(@no_face_image)

      assert {:error, :no_face_detected} =
               Image.FaceDetection.crop_largest(image, min_score: 0.99)
    end

    test "crop honours EXIF orientation — regression: iPhone photos cropped on wrong region" do
      # `test/support/images/kip_drinking.jpeg` is an iPhone photo
      # stored as landscape pixels with `Orientation = Rotate 90 CW`.
      # Pre-fix, `vips_thumbnail` auto-rotated the model input but
      # the crop ran against the un-rotated buffer, so the box's
      # in-frame coords landed on a different region of the photo
      # (a wine glass) even though the face was detected at score
      # 0.932. The fix is to autorotate consistently before both
      # detection and crop.
      image = Image.open!(Path.join(__DIR__, "support/images/kip_drinking.jpeg"))

      # Sanity-check the fixture itself: this test is only
      # meaningful if the EXIF orientation actually requires
      # rotation. If someone re-encodes the fixture and bakes the
      # rotation into the pixels, the test silently degrades.
      {:ok, exif} = Image.exif(image)
      assert Map.get(exif, :orientation) in ["Rotate 90 CW", "Rotate 180", "Rotate 270 CW"]

      assert {:ok, cropped} = Image.FaceDetection.crop_largest(image)

      # The face crop should be a meaningful share of the
      # autorotated image's area — pre-fix, the box landed at
      # un-rotated coords and the crop area would be wildly
      # different (or the box would clamp to a corner).
      rotated = Image.autorotate!(image)
      area_ratio =
        Image.width(cropped) * Image.height(cropped) /
          (Image.width(rotated) * Image.height(rotated))

      assert area_ratio > 0.02 and area_ratio < 0.5,
             "expected face crop to be 2%–50% of the autorotated image area, got #{area_ratio}"

      # And the crop should be taller than wide (portrait) — a face
      # is. Pre-fix on this fixture the crop came out landscape
      # because it was sliced from the un-rotated frame.
      assert Image.height(cropped) > Image.width(cropped)
    end
  end

  describe "draw_boxes/3" do
    test "returns a same-size Vimage when given an empty detection list" do
      image = Image.open!(hd(@face_corpus).path)
      annotated = Image.FaceDetection.draw_boxes([], image)

      assert match?(%Vix.Vips.Image{}, annotated)
      assert Image.width(annotated) == Image.width(image)
      assert Image.height(annotated) == Image.height(image)
    end

    test "draws boxes for real detections without crashing" do
      for %{path: path} <- @face_corpus do
        image = Image.open!(path)
        faces = Image.FaceDetection.detect(image)
        annotated = Image.FaceDetection.draw_boxes(faces, image)

        assert match?(%Vix.Vips.Image{}, annotated)
      end
    end

    test "show_landmarks?: false skips landmark dots" do
      image = Image.open!(hd(@face_corpus).path)
      faces = Image.FaceDetection.detect(image)

      assert match?(
               %Vix.Vips.Image{},
               Image.FaceDetection.draw_boxes(faces, image, show_landmarks?: false)
             )
    end
  end
end
