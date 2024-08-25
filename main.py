from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2


def main():
    # Read video frames
    input_video_path = "input-videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect players in all frames
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models/last.pt")
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stubs=True,
        stub_path="tracker_stubs/player_detections.pkl",
    )
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stubs=True,
        stub_path="tracker_stubs/ball_detections.pkl",
    )

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court Line Detector Model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Choose players
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints, player_detections
    )

    # Mini Court
    mini_court = MiniCourt(video_frames[0])

    # Get ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print(f"Ball shot frames: {ball_shot_frames}")

    # Draw output
    ## Draw bounding boxes around players
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    ## Draw bounding boxes around the ball
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(
        output_video_frames, court_keypoints
    )

    ## Draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(
            frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

    # Save video with player detections
    save_video(output_video_frames, "output-videos/output_video.avi")


if __name__ == "__main__":
    main()
