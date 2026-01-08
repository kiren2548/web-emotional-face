"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

type CvType = any;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);
  const [status, setStatus] = useState<string>("ยังไม่เริ่ม");
  const [emotion, setEmotion] = useState<string>("-");
  const [conf, setConf] = useState<number>(0);

  async function loadOpenCV() {
    if (typeof window === "undefined") return;

    if ((window as any).cv?.Mat) {
      cvRef.current = (window as any).cv;
      return;
    }

    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/opencv/opencv.js";
      script.async = true;

      script.onload = () => {
        const cv = (window as any).cv;
        if (!cv) return reject(new Error("OpenCV โหลดแล้วแต่ window.cv ไม่มีค่า"));

        const waitReady = () => {
          if ((window as any).cv?.Mat) {
            cvRef.current = (window as any).cv;
            resolve();
          } else {
            setTimeout(waitReady, 50);
          }
        };

        if ("onRuntimeInitialized" in cv) {
          cv.onRuntimeInitialized = () => waitReady();
        } else {
          waitReady();
        }
      };

      script.onerror = () => reject(new Error("โหลด /opencv/opencv.js ไม่สำเร็จ"));
      document.body.appendChild(script);
    });
  }
  async function loadCascade() {
    const cv = cvRef.current;
    if (!cv) throw new Error("cv ยังไม่พร้อม");

    const cascadeUrl = "/opencv/haarcascade_frontalface_default.xml";
    const res = await fetch(cascadeUrl);
    const data = new Uint8Array(await res.arrayBuffer());

    const cascadePath = "haarcascade_frontalface_default.xml";
    try { cv.FS_unlink(cascadePath); } catch {}
    cv.FS_createDataFile("/", cascadePath, data, true, false, false);

    const faceCascade = new cv.CascadeClassifier();
    const loaded = faceCascade.load(cascadePath);
    if (!loaded) throw new Error("cascade load() ไม่สำเร็จ");
    faceCascadeRef.current = faceCascade;
  }

  async function loadModel() {
    const session = await ort.InferenceSession.create(
      "/models/emotion_yolo11n_cls.onnx",
      { executionProviders: ["wasm"] }
    );
    sessionRef.current = session;

    const clsRes = await fetch("/models/classes.json");
    classesRef.current = await clsRes.json();
  }


  async function startCamera() {
    setStatus("ขอสิทธิ์กล้อง...");
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    if (!videoRef.current) return;
    videoRef.current.srcObject = stream;
    await videoRef.current.play();
    setStatus("กำลังทำงาน...");
    requestAnimationFrame(loop);
  }

  function preprocessToTensor(faceCanvas: HTMLCanvasElement) {
    const size = 64;
    const tmp = document.createElement("canvas");
    tmp.width = size; tmp.height = size;
    const ctx = tmp.getContext("2d")!;
    ctx.drawImage(faceCanvas, 0, 0, size, size);

    const imgData = ctx.getImageData(0, 0, size, size).data;
    const float = new Float32Array(1 * 3 * size * size);

    let idx = 0;
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < size * size; i++) {
        const r = imgData[i * 4 + 0] / 255;
        const g = imgData[i * 4 + 1] / 255;
        const b = imgData[i * 4 + 2] / 255;
        float[idx++] = c === 0 ? r : c === 1 ? g : b;
      }
    }

    return new ort.Tensor("float32", float, [1, 3, size, size]);
  }


  function softmax(logits: Float32Array) {
    let max = -Infinity;
    for (const v of logits) max = Math.max(max, v);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  
  async function loop() {
    const cv = cvRef.current;
    const faceCascade = faceCascadeRef.current;
    const session = sessionRef.current;
    const classes = classesRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!cv || !faceCascade || !session || !classes || !video || !canvas) {
      requestAnimationFrame(loop);
      return;
    }

    const ctx = canvas.getContext("2d")!;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const src = cv.imread(canvas);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    const faces = new cv.RectVector();
    const msize = new cv.Size(0, 0);
    faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);

    let bestRect: any = null;
    let bestArea = 0;

    for (let i = 0; i < faces.size(); i++) {
      const r = faces.get(i);
      const area = r.width * r.height;
      if (area > bestArea) { bestArea = area; bestRect = r; }
      ctx.strokeStyle = "lime";
      ctx.lineWidth = 2;
      ctx.strokeRect(r.x, r.y, r.width, r.height);
    }

    if (bestRect) {
      const faceCanvas = document.createElement("canvas");
      faceCanvas.width = bestRect.width; faceCanvas.height = bestRect.height;
      const fctx = faceCanvas.getContext("2d")!;
      fctx.drawImage(canvas, bestRect.x, bestRect.y, bestRect.width, bestRect.height, 0, 0, bestRect.width, bestRect.height);

      const input = preprocessToTensor(faceCanvas);
      const feeds: Record<string, ort.Tensor> = {};
      feeds[session.inputNames[0]] = input;

      const out = await session.run(feeds);
      const outName = session.outputNames[0];
      const logits = out[outName].data as Float32Array;
      const probs = softmax(logits);

      let maxIdx = 0;
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[maxIdx]) maxIdx = i;
      }

      setEmotion(classes[maxIdx] ?? `class_${maxIdx}`);
      setConf(probs[maxIdx] ?? 0);

      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(bestRect.x, Math.max(0, bestRect.y - 28), 220, 28);
      ctx.fillStyle = "white";
      ctx.font = "16px sans-serif";
      ctx.fillText(`${classes[maxIdx]} ${(probs[maxIdx] * 100).toFixed(1)}%`, bestRect.x + 6, bestRect.y - 8);
    }

    src.delete(); gray.delete(); faces.delete();
    requestAnimationFrame(loop);
  }

  useEffect(() => {
    (async () => {
      try {
        setStatus("กำลังโหลด OpenCV...");
        await loadOpenCV();

        setStatus("กำลังโหลด Haar cascade...");
        await loadCascade();

        setStatus("กำลังโหลดโมเดล ONNX...");
        await loadModel();

        setStatus("พร้อม เริ่มกดปุ่ม Start");
      } catch (e: any) {
        setStatus(`เริ่มต้นไม่สำเร็จ: ${e?.message ?? e}`);
      }
    })();
  }, []);

return (
  <main className="min-h-screen bg-gradient-to-br from-green-50 via-green-100 to-purple-50 text-gray-900 dark:text-purple-900 p-4 flex justify-center">
    <div className="w-full max-w-5xl space-y-5">
      <header className="space-y-1 text-center">
        <h1 className="text-3xl font-extrabold tracking-tight text-violet-400">
          Face Emotion Detection
        </h1>
        <p className="text-sm text-green-700 dark:text-purple-400">
          OpenCV + YOLO11 Classification (ONNX Runtime Web)
        </p>
      </header>

      <section className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="rounded-2xl p-4 bg-green-50/50 dark:bg-purple-100/30 border border-green-100 dark:border-purple-200 shadow-md">
          <div className="text-xs text-green-700 dark:text-purple-400 flex items-center gap-2">
            Status
          </div>
          <div className="mt-1 text-lg font-semibold text-green-600 dark:text-purple-500 truncate">
            {status}
          </div>
        </div>

        <div className="rounded-2xl p-4 bg-green-50/50 dark:bg-purple-100/30 border border-green-100 dark:border-purple-200 shadow-md">
          <div className="text-xs text-green-700 dark:text-purple-400 flex items-center gap-2">
            Emotion
          </div>
          <div className="mt-1 text-xl font-bold text-green-600 dark:text-purple-500">
            {emotion}
          </div>
        </div>

        <div className="rounded-2xl p-4 bg-green-50/50 dark:bg-purple-100/30 border border-green-100 dark:border-purple-200 shadow-md">
          <div className="text-xs text-green-700 dark:text-purple-400 flex items-center gap-2">
            Confidence
          </div>
          <div className="mt-1 text-xl font-bold text-green-600 dark:text-purple-500">
            {(conf * 100).toFixed(1)}%
          </div>
          <div className="mt-2 h-2 w-full rounded-full bg-green-100 dark:bg-purple-200 overflow-hidden">
            <div
              className="h-full bg-green-500 dark:bg-purple-500 transition-all"
              style={{ width: `${Math.min(conf * 100, 100)}%` }}
            />
          </div>
        </div>
      </section>

      <section className="flex justify-center">
        <button
          onClick={startCamera}
          className="
            flex items-center gap-2
            px-8 py-3 rounded-full font-semibold
            bg-green-500 text-white
            hover:bg-green-400 dark:bg-purple-400 dark:hover:bg-purple-500
            active:scale-95
            transition-all
            shadow-lg shadow-green-500/30 dark:shadow-purple-400/30
          "
        >
          Start Camera
        </button>
      </section>
    {/* Camera + Canvas */}
<div className="flex justify-center w-full">
  <div className="relative w-full max-w-3xl rounded-2xl shadow-lg overflow-hidden bg-green-50 dark:bg-purple-50 border border-green-100 dark:border-purple-200">
    <video ref={videoRef} className="hidden" playsInline />
    <canvas
      ref={canvasRef}
      className="w-full rounded-2xl border border-green-200 dark:border-purple-300 shadow-inner"
      style={{ aspectRatio: 'auto' }}
    />
  </div>
</div>


      <p className="text-xs text-green-700 dark:text-purple-400 text-center">
        กดปุ่ม <span className="font-medium">Start Camera</span> เพื่อขอสิทธิ์ใช้งานกล้อง
      </p>

    </div>
  </main>
);


}
