from ultralytics import YOLO
import cv2
import pickle
import pandas as pd


class BallTracker:
    """
    The BallTracker class is used to detect the tennis ball in a frame or multiple frames.
    It uses YOLO model to detect the ball.
    """

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        # List to Pandas dataframe
        df_ball_positions = (
            pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
            .interpolate()  # Interpolate missing frames
            .bfill()
        )

        return [{1: x} for x in df_ball_positions.to_numpy().tolist()]

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        # List to Pandas dataframe
        df_ball_positions = (
            pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
            .interpolate()  # Interpolate missing frames
            .bfill()
        )

        df_ball_positions["ball_hit"] = 0

        df_ball_positions["mid_y"] = (
            df_ball_positions["y1"] + df_ball_positions["y2"]
        ) / 2
        df_ball_positions["mid_y_rolling_mean"] = (
            df_ball_positions["mid_y"].rolling(10).mean()
        )
        df_ball_positions["delta_y"] = df_ball_positions["mid_y_rolling_mean"].diff()

        minimum_changes_to_detect_hit = 25
        for i in range(
            1, len(df_ball_positions) - int(minimum_changes_to_detect_hit * 1.2)
        ):
            neg_position_change = (
                df_ball_positions["delta_y"].iloc[i] > 0
                and df_ball_positions["delta_y"].iloc[i + 1] < 0
            )
            pos_position_change = (
                df_ball_positions["delta_y"].iloc[i] < 0
                and df_ball_positions["delta_y"].iloc[i + 1] > 0
            )

            if neg_position_change or pos_position_change:
                change_count = 0
                for change_frame in range(
                    i + 1, i + int(minimum_changes_to_detect_hit * 1.2) + 1
                ):
                    neg_position_change_next = (
                        df_ball_positions["delta_y"].iloc[i] > 0
                        and df_ball_positions["delta_y"].iloc[change_frame] < 0
                    )
                    pos_position_change_next = (
                        df_ball_positions["delta_y"].iloc[i] < 0
                        and df_ball_positions["delta_y"].iloc[change_frame] > 0
                    )

                    if neg_position_change and neg_position_change_next:
                        change_count += 1
                    elif pos_position_change and pos_position_change_next:
                        change_count += 1

                if change_count > minimum_changes_to_detect_hit - 1:
                    df_ball_positions["ball_hit"].iloc[i] = 1

        return df_ball_positions[df_ball_positions["ball_hit"] == 1].index.tolist()

    # This method is used to detect balls in multiple frames.
    def detect_frames(self, frames, read_from_stubs=False, stub_path=None):
        ball_detections = []

        # Read ball detections from a pickle file
        if read_from_stubs and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        # Save ball detections to a pickle file
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections  # Results of all frames

    # This method is used to detect balls in a single frame.
    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]  # [x1, y1, x2, y2]

            ball_dict[1] = result

        return ball_dict  # Results of 1 frame

    # This method is used to draw bounding boxes around players in the video frames.
    def draw_bboxes(self, video_frames, ball_detection):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detection):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    f"Ball ID: {track_id}",
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
                frame = cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
                )
            output_video_frames.append(frame)

        return output_video_frames
