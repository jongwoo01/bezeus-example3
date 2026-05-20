"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { HandLandmarker, NormalizedLandmark } from "@mediapipe/tasks-vision";

type CameraState = "idle" | "loading" | "ready" | "denied" | "unavailable" | "error";
type ViewMode = "home" | "setting";

interface LightningState {
  x: number;
  y: number;
  hue: number;
  intensity: number;
  speed: number;
  size: number;
  angle: number;
  openAmount: number;
  handVisible: boolean;
  handOpen: boolean;
}

interface LightningProps {
  hue?: number;
  xOffset?: number;
  speed?: number;
  intensity?: number;
  size?: number;
}

const initialLightning: LightningState = {
  x: 0,
  y: 0,
  hue: 220,
  intensity: 0.6,
  speed: 1.6,
  size: 2,
  angle: 0,
  openAmount: 0,
  handVisible: false,
  handOpen: false,
};

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const distance = (a: NormalizedLandmark, b: NormalizedLandmark) =>
  Math.hypot(a.x - b.x, a.y - b.y);

const mix = (from: number, to: number, amount: number) =>
  from + (to - from) * amount;

const canUseCamera = () =>
  typeof navigator !== "undefined" &&
  typeof navigator.mediaDevices?.getUserMedia === "function";

const isTfliteDelegateInfo = (value: unknown) =>
  typeof value === "string" &&
  value.includes("Created TensorFlow Lite XNNPACK delegate for CPU");

const shouldIgnoreMediapipeRuntimeLog = (args: unknown[]) =>
  args.some(isTfliteDelegateInfo);

const originalLightningSettings = {
  hue: 220,
  xOffset: 0,
  speed: 1.6,
  intensity: 0.6,
  size: 2,
};

function analyzeHand(
  landmarks: NormalizedLandmark[],
  previous?: { x: number; y: number; time: number }
) {
  const palm = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]];
  const center = palm.reduce(
    (point, landmark) => ({
      x: point.x + landmark.x / palm.length,
      y: point.y + landmark.y / palm.length,
    }),
    { x: 0, y: 0 }
  );

  const openFingers = [
    landmarks[8].y < landmarks[6].y - 0.015,
    landmarks[12].y < landmarks[10].y - 0.015,
    landmarks[16].y < landmarks[14].y - 0.015,
    landmarks[20].y < landmarks[18].y - 0.015,
  ].filter(Boolean).length;

  const thumbSpread = distance(landmarks[4], landmarks[9]);
  const fingerSpread = distance(landmarks[8], landmarks[20]);
  const openAmount = clamp((openFingers + clamp(fingerSpread * 3.5, 0, 1)) / 5, 0, 1);
  const isOpen = openFingers >= 3 && fingerSpread > 0.13 && thumbSpread > 0.11;
  const palmAngle = Math.atan2(
    landmarks[5].y - landmarks[17].y,
    landmarks[5].x - landmarks[17].x
  );

  const now = performance.now();
  const deltaSeconds = previous ? Math.max((now - previous.time) / 1000, 0.016) : 0.016;
  const movement = previous
    ? Math.hypot(center.x - previous.x, center.y - previous.y) / deltaSeconds
    : 0;

  return {
    center,
    openAmount,
    isOpen,
    palmAngle,
    movement: clamp(movement, 0, 3),
    now,
  };
}

const Lightning: React.FC<LightningProps> = ({
  hue = 230,
  xOffset = 0,
  speed = 1,
  intensity = 1,
  size = 1,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const uniformsRef = useRef({ hue, xOffset, speed, intensity, size });

  useEffect(() => {
    uniformsRef.current = { hue, xOffset, speed, intensity, size };
  }, [hue, xOffset, speed, intensity, size]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeCanvas = () => {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
    };
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const gl = canvas.getContext("webgl");
    if (!gl) {
      console.error("WebGL not supported");
      return () => {
        window.removeEventListener("resize", resizeCanvas);
      };
    }

    const vertexShaderSource = `
      attribute vec2 aPosition;
      void main() {
        gl_Position = vec4(aPosition, 0.0, 1.0);
      }
    `;

    const fragmentShaderSource = `
      precision mediump float;
      uniform vec2 iResolution;
      uniform float iTime;
      uniform float uHue;
      uniform float uXOffset;
      uniform float uSpeed;
      uniform float uIntensity;
      uniform float uSize;
      
      #define OCTAVE_COUNT 10

      // Convert HSV to RGB.
      vec3 hsv2rgb(vec3 c) {
          vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0,4.0,2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
          return c.z * mix(vec3(1.0), rgb, c.y);
      }

      float hash11(float p) {
          p = fract(p * .1031);
          p *= p + 33.33;
          p *= p + p;
          return fract(p);
      }

      float hash12(vec2 p) {
          vec3 p3 = fract(vec3(p.xyx) * .1031);
          p3 += dot(p3, p3.yzx + 33.33);
          return fract((p3.x + p3.y) * p3.z);
      }

      mat2 rotate2d(float theta) {
          float c = cos(theta);
          float s = sin(theta);
          return mat2(c, -s, s, c);
      }

      float noise(vec2 p) {
          vec2 ip = floor(p);
          vec2 fp = fract(p);
          float a = hash12(ip);
          float b = hash12(ip + vec2(1.0, 0.0));
          float c = hash12(ip + vec2(0.0, 1.0));
          float d = hash12(ip + vec2(1.0, 1.0));
          
          vec2 t = smoothstep(0.0, 1.0, fp);
          return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
      }

      float fbm(vec2 p) {
          float value = 0.0;
          float amplitude = 0.5;
          for (int i = 0; i < OCTAVE_COUNT; ++i) {
              value += amplitude * noise(p);
              p *= rotate2d(0.45);
              p *= 2.0;
              amplitude *= 0.5;
          }
          return value;
      }

      void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
          // Normalized pixel coordinates.
          vec2 uv = fragCoord / iResolution.xy;
          uv = 2.0 * uv - 1.0;
          uv.x *= iResolution.x / iResolution.y;
          // Apply horizontal offset.
          uv.x += uXOffset;
          
          // Adjust uv based on size and animate with speed.
          uv += 2.0 * fbm(uv * uSize + 0.8 * iTime * uSpeed) - 1.0;
          
          float dist = abs(uv.x);
          // Compute base color using hue.
          vec3 baseColor = hsv2rgb(vec3(uHue / 360.0, 0.7, 0.8));
          // Compute color with intensity and speed affecting time.
          vec3 col = baseColor * pow(mix(0.0, 0.07, hash11(iTime * uSpeed)) / dist, 1.0) * uIntensity;
          col = pow(col, vec3(1.0));
          fragColor = vec4(col, 1.0);
      }

      void main() {
          mainImage(gl_FragColor, gl_FragCoord.xy);
      }
    `;

    const compileShader = (source: string, type: number): WebGLShader | null => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error("Shader compile error:", gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const vertexShader = compileShader(vertexShaderSource, gl.VERTEX_SHADER);
    const fragmentShader = compileShader(fragmentShaderSource, gl.FRAGMENT_SHADER);
    if (!vertexShader || !fragmentShader) {
      return () => {
        window.removeEventListener("resize", resizeCanvas);
      };
    }

    const program = gl.createProgram();
    if (!program) {
      return () => {
        window.removeEventListener("resize", resizeCanvas);
      };
    }

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("Program linking error:", gl.getProgramInfoLog(program));
      return () => {
        window.removeEventListener("resize", resizeCanvas);
      };
    }
    gl.useProgram(program);

    const vertices = new Float32Array([
      -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1,
    ]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    const aPosition = gl.getAttribLocation(program, "aPosition");
    gl.enableVertexAttribArray(aPosition);
    gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);

    const iResolutionLocation = gl.getUniformLocation(program, "iResolution");
    const iTimeLocation = gl.getUniformLocation(program, "iTime");
    const uHueLocation = gl.getUniformLocation(program, "uHue");
    const uXOffsetLocation = gl.getUniformLocation(program, "uXOffset");
    const uSpeedLocation = gl.getUniformLocation(program, "uSpeed");
    const uIntensityLocation = gl.getUniformLocation(program, "uIntensity");
    const uSizeLocation = gl.getUniformLocation(program, "uSize");

    let animationFrame = 0;
    const startTime = performance.now();
    const render = () => {
      resizeCanvas();
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(iResolutionLocation, canvas.width, canvas.height);
      const currentTime = performance.now();
      const current = uniformsRef.current;
      gl.uniform1f(iTimeLocation, (currentTime - startTime) / 1000.0);
      gl.uniform1f(uHueLocation, current.hue);
      gl.uniform1f(uXOffsetLocation, current.xOffset);
      gl.uniform1f(uSpeedLocation, current.speed);
      gl.uniform1f(uIntensityLocation, current.intensity);
      gl.uniform1f(uSizeLocation, current.size);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      animationFrame = requestAnimationFrame(render);
    };
    animationFrame = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animationFrame);
      window.removeEventListener("resize", resizeCanvas);
      gl.deleteBuffer(vertexBuffer);
      gl.deleteProgram(program);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
    };
  }, []);

  return <canvas ref={canvasRef} className="w-full h-full relative" />;
};

const statusCopy: Record<CameraState, string> = {
  idle: "Camera is waiting",
  loading: "Opening camera",
  ready: "Camera and hand tracking are live",
  denied: "Camera permission was denied",
  unavailable: "Camera is unavailable",
  error: "Camera could not start",
};

const gestureCards = [
  {
    label: "Fist",
    title: "Hold power",
    body: "Scene energy drops and lightning output stops when your hand closes.",
  },
  {
    label: "Open Hand",
    title: "Summon thunder",
    body: "Open your palm wide and the lightning blooms around your hand position.",
  },
  {
    label: "Move / Rotate",
    title: "Shape the strike",
    body: "Motion and palm angle shift the lightning position, color, and intensity.",
  },
];

export const HeroSection: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const detectionFrameRef = useRef<number | null>(null);
  const previousHandRef = useRef<{ x: number; y: number; time: number } | undefined>(
    undefined
  );
  const lastVideoTimeRef = useRef(-1);

  const [view, setView] = useState<ViewMode>("home");
  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [showPrepare, setShowPrepare] = useState(true);
  const [lightning, setLightning] = useState<LightningState>(initialLightning);
  const [streamActive, setStreamActive] = useState(false);

  useEffect(() => {
    const originalError = console.error;
    const originalInfo = console.info;
    const originalWarn = console.warn;

    console.error = (...args: Parameters<Console["error"]>) => {
      if (shouldIgnoreMediapipeRuntimeLog(args)) return;
      originalError(...args);
    };

    console.info = (...args: Parameters<Console["info"]>) => {
      if (shouldIgnoreMediapipeRuntimeLog(args)) return;
      originalInfo(...args);
    };

    console.warn = (...args: Parameters<Console["warn"]>) => {
      if (shouldIgnoreMediapipeRuntimeLog(args)) return;
      originalWarn(...args);
    };

    return () => {
      console.error = originalError;
      console.info = originalInfo;
      console.warn = originalWarn;
    };
  }, []);

  const stopDetection = useCallback(() => {
    if (detectionFrameRef.current) {
      cancelAnimationFrame(detectionFrameRef.current);
      detectionFrameRef.current = null;
    }
  }, []);

  const stopCamera = useCallback(() => {
    stopDetection();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setStreamActive(false);
    landmarkerRef.current?.close();
    landmarkerRef.current = null;
    previousHandRef.current = undefined;
  }, [stopDetection]);

  const startDetection = useCallback(() => {
    const detect = () => {
      const video = videoRef.current;
      const landmarker = landmarkerRef.current;

      if (video && landmarker && video.readyState >= 2) {
        if (video.currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = video.currentTime;
          const result = landmarker.detectForVideo(video, performance.now());
          const landmarks = result.landmarks[0];

          if (landmarks) {
            const analyzed = analyzeHand(landmarks, previousHandRef.current);
            const mirroredX = 1 - analyzed.center.x;
            const targetX = clamp((mirroredX - 0.5) * 1.72, -0.86, 0.86);
            const targetY = clamp((0.5 - analyzed.center.y) * 1.28, -0.64, 0.64);
            previousHandRef.current = {
              x: analyzed.center.x,
              y: analyzed.center.y,
              time: analyzed.now,
            };

            setLightning((current) => ({
              x: mix(current.x, targetX, 0.36),
              y: mix(current.y, targetY, 0.28),
              hue: originalLightningSettings.hue,
              intensity: originalLightningSettings.intensity,
              speed: originalLightningSettings.speed,
              size: originalLightningSettings.size,
              angle: mix(current.angle, analyzed.palmAngle * 0.42, 0.2),
              openAmount: mix(current.openAmount, analyzed.openAmount, 0.26),
              handVisible: true,
              handOpen: analyzed.isOpen,
            }));
          } else {
            previousHandRef.current = undefined;
            setLightning((current) => ({
              ...current,
              openAmount: mix(current.openAmount, 0, 0.12),
              handVisible: false,
              handOpen: false,
            }));
          }
        }
      }

      detectionFrameRef.current = requestAnimationFrame(detect);
    };

    stopDetection();
    detect();
  }, [stopDetection]);

  const enableCamera = useCallback(async () => {
    if (cameraState === "loading") return;

    setShowPrepare(false);
    setCameraState("loading");

    try {
      if (!canUseCamera()) {
        setCameraState("unavailable");
        setShowPrepare(true);
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = stream;
      setStreamActive(true);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      if (!landmarkerRef.current) {
        const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision");
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
        );

        landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.45,
          minHandPresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });
      }

      setCameraState("ready");
      startDetection();
    } catch (error) {
      stopCamera();
      const name = error instanceof DOMException ? error.name : "";
      setCameraState(name === "NotAllowedError" || name === "PermissionDeniedError" ? "denied" : "error");
      setShowPrepare(true);
    }
  }, [cameraState, startDetection, stopCamera]);

  const refreshStatus = useCallback(() => {
    if (streamRef.current?.active) {
      setStreamActive(true);
      setCameraState("ready");
      return;
    }

    setStreamActive(false);
    setCameraState(canUseCamera() ? "idle" : "unavailable");
  }, []);

  useEffect(() => {
    refreshStatus();

    return () => {
      stopCamera();
    };
  }, [refreshStatus, stopCamera]);

  const handLeft = `${(lightning.x / 1.72 + 0.5) * 100}%`;
  const handTop = `${(0.5 - lightning.y / 1.28) * 100}%`;
  const isCameraReady = cameraState === "ready";

  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-black text-white">
      <video
        ref={videoRef}
        playsInline
        muted
        className="pointer-events-none absolute h-px w-px opacity-0"
      />

      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-[linear-gradient(180deg,#050505_0%,#000_44%,#05040a_100%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.035)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.025)_1px,transparent_1px)] bg-[size:64px_64px] opacity-25" />
        <div className="absolute inset-x-0 top-0 h-36 border-b border-white/10 bg-white/[0.025] backdrop-blur-[2px]" />
        <motion.div
          className="absolute -inset-y-[18%] inset-x-0"
          animate={{
            opacity: lightning.handOpen ? 1 : 0,
            y: `${-lightning.y * 18}vh`,
          }}
          transition={{ duration: 0.28, ease: "easeOut" }}
        >
          <Lightning
            hue={originalLightningSettings.hue}
            xOffset={originalLightningSettings.xOffset - lightning.x}
            speed={originalLightningSettings.speed}
            intensity={originalLightningSettings.intensity}
            size={originalLightningSettings.size}
          />
        </motion.div>
      </div>

      <motion.div
        className="pointer-events-none absolute z-10 h-48 w-48 -translate-x-1/2 -translate-y-1/2 rounded-full border border-cyan-200/15 bg-cyan-200/[0.025]"
        style={{
          left: handLeft,
          top: handTop,
          boxShadow: `0 0 ${36 + lightning.openAmount * 80}px rgba(92, 203, 255, ${
            lightning.handOpen ? 0.28 : 0.05
          })`,
          opacity: lightning.handVisible ? 0.42 : 0,
          scale: 0.78 + lightning.openAmount * 0.32,
        }}
        animate={{ rotate: lightning.angle * 40 }}
        transition={{ type: "spring", stiffness: 160, damping: 28 }}
      >
        <div className="absolute left-1/2 top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-100/45 shadow-[0_0_22px_rgba(125,230,255,0.55)]" />
        <div className="absolute inset-8 rounded-full border border-white/10 blur-[1px]" />
      </motion.div>

      <header className="absolute left-0 right-0 top-0 z-30 px-4 py-5 sm:px-8">
        <nav className="mx-auto flex max-w-6xl items-center justify-between rounded-full border border-white/15 bg-black/35 px-4 py-3 shadow-[0_0_40px_rgba(80,170,255,0.08)] backdrop-blur-xl sm:px-5">
          <button
            className="flex items-center gap-3 text-left"
            onClick={() => setView("home")}
            aria-label="Go home"
          >
            <span className="grid h-9 w-9 place-items-center rounded-full border border-cyan-200/25 bg-white/[0.04] text-cyan-200 shadow-[0_0_18px_rgba(90,200,255,0.18)]">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path
                  d="M13 2L4 13H11L9 22L20 9H13L13 2Z"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinejoin="round"
                />
              </svg>
            </span>
            <span className="text-sm font-semibold tracking-[0.24em] text-white">BeZeus</span>
          </button>

          <div className="flex items-center gap-1 text-sm text-white/68 sm:gap-3">
            <button
              className="rounded-full px-3 py-2 transition hover:bg-white/10 hover:text-white"
              onClick={() => setView("home")}
            >
              Home
            </button>
            <button
              className="rounded-full px-3 py-2 transition hover:bg-white/10 hover:text-white"
              onClick={() => setView("setting")}
            >
              Setting
            </button>
            <a
              className="rounded-full px-3 py-2 transition hover:bg-white/10 hover:text-white"
              href="https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js"
              target="_blank"
              rel="noreferrer"
            >
              Docs
            </a>
          </div>
        </nav>
      </header>

      <main className="relative z-20 min-h-screen px-5 pb-8 pt-28 sm:px-8">
        <AnimatePresence mode="wait">
          {view === "home" ? (
            <motion.section
              key="home"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }}
              transition={{ duration: 0.35 }}
              className="mx-auto flex min-h-[calc(100vh-8rem)] max-w-6xl flex-col items-center justify-center text-center"
            >
              <div className="mb-5 rounded-full border border-cyan-200/20 bg-cyan-200/[0.06] px-4 py-2 text-xs font-medium uppercase tracking-[0.26em] text-cyan-100/90">
                Live Thunder Interface
              </div>
              <h1 className="max-w-4xl text-6xl font-semibold leading-none tracking-normal text-white sm:text-8xl md:text-9xl">
                Be{" "}
                <span className="bg-gradient-to-r from-cyan-200 via-blue-400 to-violet-400 bg-clip-text text-transparent">
                  Zeus
                </span>
              </h1>
              <p className="mt-7 max-w-2xl text-xl text-white/72 sm:text-2xl">
                The power of the Thunder, in the palm of your hand.
              </p>
              <p className="mt-5 max-w-2xl text-base text-white/62">
                손을 카메라 앞에 활짝 펴서 번개를 생성해보세요.
              </p>
              <p className="mt-2 max-w-2xl text-sm text-white/45">
                Open your hand wide in front of the camera to summon lightning.
              </p>

              <div className="mt-9 flex flex-col items-center gap-3 sm:flex-row">
                <motion.button
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.98 }}
                  className="rounded-full border border-cyan-200/40 bg-cyan-200 px-6 py-3 text-sm font-semibold text-black shadow-[0_0_34px_rgba(80,202,255,0.24)] transition hover:bg-white"
                  onClick={enableCamera}
                  disabled={cameraState === "loading"}
                >
                  {cameraState === "loading" ? "Preparing Camera..." : "Enable Camera to be Zeus"}
                </motion.button>
                <button
                  className="rounded-full border border-white/12 px-6 py-3 text-sm text-white/72 transition hover:border-white/30 hover:bg-white/[0.08] hover:text-white"
                  onClick={() => setView("setting")}
                >
                  Camera Setting
                </button>
              </div>

              <div className="mt-8 flex items-center gap-3 rounded-full border border-white/10 bg-white/[0.035] px-4 py-2 text-xs text-white/55 backdrop-blur-md">
                <span
                  className={`h-2 w-2 rounded-full ${
                    isCameraReady ? "bg-cyan-300 shadow-[0_0_14px_#67e8f9]" : "bg-white/30"
                  }`}
                />
                {statusCopy[cameraState]}
                {isCameraReady && (
                  <span className="text-white/35">
                    {lightning.handOpen ? "Open hand detected" : "Waiting for open palm"}
                  </span>
                )}
              </div>
            </motion.section>
          ) : (
            <motion.section
              key="setting"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }}
              transition={{ duration: 0.35 }}
              className="mx-auto flex min-h-[calc(100vh-8rem)] max-w-6xl flex-col justify-center"
            >
              <div className="mb-4 flex flex-wrap items-center gap-3">
                <span className="rounded-full border border-white/12 bg-white/[0.04] px-3 py-1 text-xs text-white/55">
                  /setting
                </span>
                <span className="text-sm text-cyan-100/70">Camera Control Center</span>
              </div>
              <h2 className="max-w-3xl text-5xl font-semibold tracking-normal text-white sm:text-7xl">
                Camera{" "}
                <span className="bg-gradient-to-r from-cyan-200 via-blue-400 to-violet-400 bg-clip-text text-transparent">
                  Setting
                </span>
              </h2>

              <div className="mt-10 grid gap-5 lg:grid-cols-[1fr_1.2fr]">
                <div className="rounded-[28px] border border-white/12 bg-white/[0.045] p-6 backdrop-blur-xl">
                  <p className="text-xs uppercase tracking-[0.26em] text-cyan-100/55">
                    Camera State
                  </p>
                  <h3 className="mt-4 text-2xl font-semibold text-white">
                    {statusCopy[cameraState]}
                  </h3>
                  <dl className="mt-6 space-y-4 text-sm">
                    <div className="flex justify-between gap-4 border-b border-white/10 pb-3">
                      <dt className="text-white/45">Permission</dt>
                      <dd className="text-white/80">{cameraState}</dd>
                    </div>
                    <div className="flex justify-between gap-4 border-b border-white/10 pb-3">
                      <dt className="text-white/45">Stream</dt>
                      <dd className="text-white/80">
                        {streamActive ? "active" : "inactive"}
                      </dd>
                    </div>
                    <div className="flex justify-between gap-4">
                      <dt className="text-white/45">Gesture</dt>
                      <dd className="text-white/80">
                        {lightning.handOpen ? "open hand" : lightning.handVisible ? "hand closed" : "not detected"}
                      </dd>
                    </div>
                  </dl>
                  <div className="mt-7 flex flex-wrap gap-3">
                    <button
                      className="rounded-full bg-white px-4 py-2 text-sm font-medium text-black transition hover:bg-cyan-100"
                      onClick={enableCamera}
                    >
                      Enable Camera
                    </button>
                    <button
                      className="rounded-full border border-white/12 px-4 py-2 text-sm text-white/70 transition hover:bg-white/10 hover:text-white"
                      onClick={refreshStatus}
                    >
                      Refresh Status
                    </button>
                    <button
                      className="rounded-full border border-white/12 px-4 py-2 text-sm text-white/70 transition hover:bg-white/10 hover:text-white"
                      onClick={() => setView("home")}
                    >
                      Back Home
                    </button>
                  </div>
                </div>

                <div className="grid gap-4 sm:grid-cols-3">
                  {gestureCards.map((card) => (
                    <div
                      key={card.label}
                      className="rounded-[24px] border border-white/12 bg-black/35 p-5 backdrop-blur-xl"
                    >
                      <p className="text-xs uppercase tracking-[0.22em] text-cyan-100/55">
                        {card.label}
                      </p>
                      <h3 className="mt-4 text-xl font-semibold text-white">{card.title}</h3>
                      <p className="mt-3 text-sm leading-6 text-white/55">{card.body}</p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.section>
          )}
        </AnimatePresence>
      </main>

      <AnimatePresence>
        {showPrepare && view === "home" && cameraState !== "ready" && (
          <motion.div
            className="fixed inset-0 z-40 grid place-items-center bg-black/48 px-5 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.96, y: 18 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: 18 }}
              className="w-full max-w-md rounded-[28px] border border-white/14 bg-zinc-950/78 p-6 text-left shadow-[0_30px_100px_rgba(0,0,0,0.55)] backdrop-blur-2xl"
            >
              <p className="text-xs uppercase tracking-[0.28em] text-cyan-100/55">Prepare</p>
              <h2 className="mt-4 text-3xl font-semibold text-white">
                Prepare to channel thunder.
              </h2>
              <p className="mt-4 text-sm leading-6 text-white/58">
                Enable your camera to start hand tracking. Open your palm in front of the
                lens and BeZeus will bind the lightning to your hand.
              </p>
              {(cameraState === "denied" || cameraState === "unavailable" || cameraState === "error") && (
                <p className="mt-4 rounded-2xl border border-rose-300/15 bg-rose-400/10 p-3 text-sm text-rose-100/80">
                  {cameraState === "denied"
                    ? "Camera permission is blocked. Update browser permissions and try again."
                    : "The camera could not be started on this device or browser."}
                </p>
              )}
              <div className="mt-6 flex flex-col gap-3 sm:flex-row">
                <button
                  className="rounded-full bg-white px-5 py-3 text-sm font-semibold text-black transition hover:bg-cyan-100"
                  onClick={enableCamera}
                  disabled={cameraState === "loading"}
                >
                  {cameraState === "loading" ? "Preparing..." : "Enable Camera to be Zeus"}
                </button>
                <button
                  className="rounded-full border border-white/12 px-5 py-3 text-sm text-white/68 transition hover:bg-white/10 hover:text-white"
                  onClick={() => setShowPrepare(false)}
                >
                  Not Now
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const DemoOne = () => {
  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-black">
      <HeroSection />
    </div>
  );
};

export { DemoOne };
