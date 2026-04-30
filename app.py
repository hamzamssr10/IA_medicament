"""
4-Camera WebRTC Streaming Server
Requirements:
    pip install aiohttp aiortc opencv-python av aiohttp_cors

Usage:
    python cam_server.py

    - Opens cameras at indices 0, 1, 2, 3 (USB / built-in webcams).
    - Change CAM_INDICES below to use different devices or RTSP URLs.
    - Visit http://localhost:8080 in a browser.
"""

import asyncio
import json
import logging
import uuid
from fractions import Fraction

import cv2
import av
import numpy as np
from aiohttp import web
from aiohttp_cors import setup as cors_setup, ResourceOptions
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cam_server")

# ── Configuration ────────────────────────────────────────────────────────────
CAM_INDICES = [1, 0, 2, 3]   # Camera device indices (int) or RTSP URLs (str)
HOST        = "0.0.0.0"
PORT        = 8080
# ─────────────────────────────────────────────────────────────────────────────

relay     = MediaRelay()
pcs: set  = set()


# ── Custom video track that reads from OpenCV ─────────────────────────────────
class CameraTrack(VideoStreamTrack):
    """
    Wraps an OpenCV VideoCapture as an aiortc VideoStreamTrack.
    Falls back to a coloured test pattern if the camera cannot be opened.
    """
    kind = "video"

    def __init__(self, cam_id, width=640, height=480, fps=30):
        super().__init__()
        self.cam_id  = cam_id
        self.width   = width
        self.height  = height
        self.fps     = fps
        self._cap    = cv2.VideoCapture(cam_id)
        self._frame  = 0

        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS,          fps)
            logger.info("Camera %s opened.", cam_id)
        else:
            logger.warning("Camera %s NOT available – using test pattern.", cam_id)

    # colour test pattern when no real camera is present
    def _test_frame(self):
        colours = [(220, 60, 60), (60, 180, 60), (60, 60, 220), (180, 180, 60)]
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        colour = colours[self.cam_id % len(colours)] if isinstance(self.cam_id, int) else (120, 120, 120)
        img[:] = colour
        label = f"CAM {self.cam_id}  frame {self._frame}"
        cv2.putText(img, label, (20, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        if self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                img = self._test_frame()
        else:
            img = self._test_frame()

        self._frame += 1
        video_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        video_frame.pts      = pts
        video_frame.time_base = time_base
        return video_frame

    def __del__(self):
        if self._cap and self._cap.isOpened():
            self._cap.release()


# One shared track per camera (relayed to N peer connections)
_cam_tracks = {idx: relay.subscribe(CameraTrack(idx)) for idx in CAM_INDICES}


# ── HTTP handlers ─────────────────────────────────────────────────────────────
async def index(request):
    with open("index.html", "r", encoding="utf-8") as f:
        return web.Response(content_type="text/html", text=f.read())


async def offer(request):
    """Handle a WebRTC offer for a single camera stream."""
    params   = await request.json()
    cam_id   = int(params.get("cam_id", CAM_INDICES[0]))
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state():
        logger.info("PC[%s] state → %s", cam_id, pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    track = _cam_tracks.get(cam_id)
    if track is None:
        return web.Response(status=400, text=f"Unknown cam_id: {cam_id}")

    pc.addTrack(track)
    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )


async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


# ── App setup ─────────────────────────────────────────────────────────────────
app = web.Application()
app.on_shutdown.append(on_shutdown)
app.router.add_get("/",       index)
app.router.add_post("/offer", offer)

cors = cors_setup(app, defaults={
    "*": ResourceOptions(allow_credentials=True, expose_headers="*",
                         allow_headers="*", allow_methods=["POST", "GET"])
})
for route in list(app.router.routes()):
    cors.add(route)

if __name__ == "__main__":
    print(f"\n  ✦ Camera stream server running at http://localhost:{PORT}\n")
    web.run_app(app, host=HOST, port=PORT)