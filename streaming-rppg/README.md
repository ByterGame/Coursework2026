# Streaming rPPG Monitor

Browser-based demo for live remote photoplethysmography from a webcam stream.

## Run

```bash
python server.py
```

Open:

```text
http://localhost:10000
```

Camera access requires `localhost` or HTTPS in modern browsers.

## Pipeline

The app captures webcam frames, detects facial landmarks with MediaPipe, tracks a forehead ROI, extracts RGB traces, estimates pulse candidates with POS and CHROM signals, filters the signal into the expected heart-rate band, and displays a stabilized BPM estimate with lightweight signal-quality diagnostics.

## Files

- `index.html` - application markup.
- `styles.css` - responsive interface styling.
- `app.js` - camera, MediaPipe, ROI tracking, and main processing loop.
- `signal.js` - rPPG signal processing and BPM candidate selection.
- `ui.js` - status widgets, waveform rendering, and display helpers.
- `server.py` - small static server for local runs and deployment.
