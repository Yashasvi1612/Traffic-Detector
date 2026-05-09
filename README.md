# 🚦 Real-Time Crowd & Traffic Monitoring System

An AI-powered real-time detection system that monitors people and vehicles using **YOLOv8** and **DeepSORT tracking**, with live analytics, crowd alerts, and a web-based dashboard built with **Gradio**.

---

## 🚀 Features

- 👤 **Real-time People & Car Detection** using YOLOv8n
- 🔢 **Unique ID Tracking** per person/car using DeepSORT (no double counting)
- ⚠️ **Crowd Alert System** — audio beep when people exceed threshold
- 📊 **Live Stats Dashboard** — people count, car count, alert status
- 📝 **CSV Data Logging** — timestamped detection logs saved automatically
- 📈 **Analytics Dashboard** — visualize people/car counts over time with Plotly
- 🌐 **Gradio Web UI** — clean browser-based interface, no OpenCV window needed
- ⚡ **FPS Counter** — real-time performance monitoring

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| YOLOv8 (Ultralytics) | Object detection |
| DeepSORT | Multi-object tracking |
| OpenCV | Frame capture & processing |
| Gradio | Web UI & deployment |
| Plotly | Analytics charts |
| Pandas | CSV data processing |
| Python | Core language |
