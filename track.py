import pims
import trackpy as tp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---- CONFIG ----
VIDEO_PATH   = "sample video.mp4"  # change if needed
DIAMETER     = 21                  # approximate ball diameter in pixels
MINMASS      = 300                 # filter out noise; tune this
SEARCH_RANGE = 20                  # max displacement (pixels) per frame
MEMORY       = 3                   # how many frames a particle can disappear for


def process_video():
    print("[INFO] Loading video...")

    @pims.pipeline
    def to_gray(frame):
        import cv2
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame

    frames = to_gray(pims.Video(VIDEO_PATH))

    print("[INFO] Detecting features with trackpy.batch()...")
    f = tp.batch(
        frames,
        diameter=DIAMETER,
        minmass=MINMASS,
        invert=True
    )

    print("[INFO] Linking features into trajectories...")
    t = tp.link(f, search_range=SEARCH_RANGE, memory=MEMORY)

    t.to_csv("track_results.csv", index=False)
    print("[INFO] Saved to track_results.csv")

    return t



def animate_tracks(tracks):
    """
    Play the video back and overlay the tracked positions frame by frame.
    This gives a real-time style visualization of the tracking result.
    """
    print("[INFO] Preparing animation...")
    frames = pims.Video(VIDEO_PATH)

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Trackpy Ball Tracking")

    # Show first frame
    first_frame = frames[0]
    im = ax.imshow(first_frame, cmap="gray")

    # Scatter for particle positions (empty at init)
    scat = ax.scatter([], [], s=80, edgecolors="red",
                      facecolors="none", linewidths=1.5)

    ax.set_title("Ball tracking (Trackpy)")
    ax.set_axis_off()

    # If you know there's only one ball, you can filter to a single particle ID
    # e.g., find the longest trajectory:
    particle_counts = tracks["particle"].value_counts()
    main_particle_id = particle_counts.index[0]
    main_track = tracks[tracks["particle"] == main_particle_id].copy()

    def init():
        im.set_data(frames[0])
        scat.set_offsets([])
        return im, scat

    def update(frame_idx):
        """Update image and scatter for frame_idx."""
        frame = frames[frame_idx]
        im.set_data(frame)

        # Select detections corresponding to this frame
        pts = main_track[main_track["frame"] == frame_idx]

        if len(pts) > 0:
            xy = pts[["x", "y"]].values
            scat.set_offsets(xy)
        else:
            scat.set_offsets([])

        ax.set_xlabel(f"Frame: {frame_idx}")
        return im, scat

    # Change interval (ms) to match your desired playback speed
    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=40,   # ~25 fps
        blit=True
    )

    print("[INFO] Showing animation window...")
    plt.show()


def main():
    tracks = process_video()
    animate_tracks(tracks)


if __name__ == "__main__":   # 🔴 important for multiprocessing on macOS / Windows
    main()
